"""Two-stage training, checkpointing, and evaluation for the fusion model."""

from __future__ import annotations

import argparse
import copy
import json
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from torch.optim import AdamW
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

from .dataset import BATCH_SIZE, MAX_LEN, PreparedData, make_loaders, prepare_datasets
from .model import DEFAULT_TEXT_ENCODER, BioClinicalMultiModalFusion

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = REPO_ROOT / "data" / "synthetic_patient_records_5000_range_anonymized.csv"
DEFAULT_MODELS_DIR = REPO_ROOT / "models"


@dataclass
class TrainConfig:
    warmup_epochs: int = 3
    finetune_epochs: int = 5
    batch_size: int = BATCH_SIZE
    max_len: int = MAX_LEN
    bert_lr: float = 1e-5
    head_lr: float = 2e-4
    warmup_head_lr: float = 1e-3
    weight_decay: float = 0.01
    hidden_dim: int = 64
    dropout: float = 0.3
    seed: int = 42


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def load_tokenizer(name: str):
    try:
        return AutoTokenizer.from_pretrained(name), name
    except Exception as exc:  # noqa: BLE001 - offline fallback
        fallback = "bert-base-uncased"
        print(f"Could not load {name} ({exc}); falling back to {fallback}")
        return AutoTokenizer.from_pretrained(fallback), fallback


def run_epoch(model, loader, optimizer, scheduler, criterion, device) -> float:
    model.train()
    total_loss = 0.0

    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        tabular = batch["tabular"].to(device)
        labels = batch["label"].to(device)

        optimizer.zero_grad(set_to_none=True)
        logits = model(input_ids=input_ids, attention_mask=attention_mask, tabular=tabular)
        loss = criterion(logits, labels)
        loss.backward()
        nn.utils.clip_grad_norm_(
            (p for p in model.parameters() if p.requires_grad), max_norm=1.0
        )
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        total_loss += loss.item() * labels.size(0)

    return total_loss / len(loader.dataset)


@torch.no_grad()
def predict(model, loader, criterion, device) -> tuple[float, np.ndarray, np.ndarray]:
    model.eval()
    total_loss = 0.0
    probs: list[np.ndarray] = []
    labels: list[np.ndarray] = []

    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        tabular = batch["tabular"].to(device)
        batch_labels = batch["label"].to(device)

        logits = model(input_ids=input_ids, attention_mask=attention_mask, tabular=tabular)
        total_loss += criterion(logits, batch_labels).item() * batch_labels.size(0)
        probs.append(torch.sigmoid(logits).cpu().numpy())
        labels.append(batch_labels.cpu().numpy().astype(int))

    mean_loss = total_loss / len(loader.dataset)
    return mean_loss, np.concatenate(probs), np.concatenate(labels)


def best_f1_threshold(labels: np.ndarray, probs: np.ndarray) -> tuple[float, float]:
    precision, recall, thresholds = precision_recall_curve(labels, probs)
    f1 = 2 * precision * recall / np.clip(precision + recall, 1e-12, None)
    best_index = int(np.argmax(f1[:-1]))
    return float(thresholds[best_index]), float(f1[best_index])


def train(prepared: PreparedData, tokenizer, tokenizer_name: str, config: TrainConfig,
          device: torch.device, models_dir: Path) -> dict:
    splits = make_loaders(
        prepared,
        tokenizer,
        batch_size=config.batch_size,
        max_len=config.max_len,
        seed=config.seed,
    )

    criterion = nn.BCEWithLogitsLoss(pos_weight=splits.pos_weight.to(device))
    model = BioClinicalMultiModalFusion(
        num_tabular_features=prepared.X_train.shape[1],
        bert_model_name=tokenizer_name,
        hidden_dim=config.hidden_dim,
        dropout_rate=config.dropout,
    ).to(device)

    checkpoint_path = models_dir / "best_multimodal_model.pt"
    state: dict = {"best_val_auc": -float("inf"), "best_state": None, "best_epoch": ""}

    def evaluate_and_track(stage_label: str) -> None:
        val_loss, val_probs, val_labels = predict(model, splits.val_loader, criterion, device)
        val_auc = roc_auc_score(val_labels, val_probs)
        val_acc = accuracy_score(val_labels, (val_probs >= 0.5).astype(int))
        print(
            f"{stage_label} | val_loss {val_loss:.4f} | val_acc {val_acc * 100:.2f}% "
            f"| val_auc {val_auc:.4f}"
        )
        if np.isfinite(val_auc) and val_auc > state["best_val_auc"]:
            state["best_val_auc"] = val_auc
            state["best_state"] = copy.deepcopy(model.state_dict())
            state["best_epoch"] = stage_label
            print(f"{stage_label} | new best checkpoint (val_auc {val_auc:.4f})")

    # Stage 1: frozen text encoder, train new heads only.
    print("\nSTAGE 1: frozen text encoder warmup")
    for param in model.bert.parameters():
        param.requires_grad = False

    optimizer = AdamW(
        [
            {"params": model.tabular_encoder.parameters(), "lr": config.warmup_head_lr,
             "weight_decay": config.weight_decay},
            {"params": model.classifier.parameters(), "lr": config.warmup_head_lr,
             "weight_decay": config.weight_decay},
        ]
    )

    for epoch in range(1, config.warmup_epochs + 1):
        start = time.time()
        model.train()
        model.bert.eval()
        train_loss = run_epoch(model, splits.train_loader, optimizer, None, criterion, device)
        print(f"Warmup [{epoch}/{config.warmup_epochs}] ({time.time() - start:.1f}s) "
              f"| train_loss {train_loss:.4f}")
        evaluate_and_track(f"Warmup_epoch_{epoch}")

    # Stage 2: joint fine-tuning with a linear warmup schedule.
    print("\nSTAGE 2: joint multimodal fine-tuning")
    for param in model.bert.parameters():
        param.requires_grad = True

    optimizer = AdamW(
        [
            {"params": model.bert.parameters(), "lr": config.bert_lr,
             "weight_decay": config.weight_decay},
            {"params": model.tabular_encoder.parameters(), "lr": config.head_lr,
             "weight_decay": config.weight_decay},
            {"params": model.classifier.parameters(), "lr": config.head_lr,
             "weight_decay": config.weight_decay},
        ]
    )
    total_steps = len(splits.train_loader) * config.finetune_epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_steps * 0.1),
        num_training_steps=total_steps,
    )

    for epoch in range(1, config.finetune_epochs + 1):
        start = time.time()
        train_loss = run_epoch(model, splits.train_loader, optimizer, scheduler, criterion, device)
        print(f"Fine-tuning [{epoch}/{config.finetune_epochs}] ({time.time() - start:.1f}s) "
              f"| train_loss {train_loss:.4f}")
        evaluate_and_track(f"Fine-tuning_epoch_{epoch}")

    if state["best_state"] is None:
        raise RuntimeError("No valid checkpoint was produced.")

    model.load_state_dict(state["best_state"])

    # Pick the decision threshold on the validation split, never on test.
    _, val_probs, val_labels = predict(model, splits.val_loader, criterion, device)
    threshold, val_f1 = best_f1_threshold(val_labels, val_probs)

    prepared.preprocessor.save(models_dir / "preprocessor.json")

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "best_val_auc": state["best_val_auc"],
        "best_val_f1": val_f1,
        "decision_threshold": threshold,
        "stage_epoch": state["best_epoch"],
        "tokenizer_name": tokenizer_name,
        "max_len": config.max_len,
        "feature_dim": prepared.X_train.shape[1],
        "numeric_cols": prepared.preprocessor.numeric_cols,
        "categorical_cols": prepared.preprocessor.categorical_cols,
        "tabular_feature_names": prepared.feature_names,
        "pos_weight": float(splits.pos_weight),
        "config": asdict(config),
    }
    torch.save(checkpoint, checkpoint_path)
    print(f"\nSaved checkpoint to {checkpoint_path}")
    print(f"Best val ROC-AUC: {state['best_val_auc']:.4f} | "
          f"F1-optimal threshold: {threshold:.3f} (val F1 {val_f1:.4f})")

    return {"model": model, "splits": splits, "criterion": criterion,
            "checkpoint": checkpoint}


def evaluate_test(model, splits, criterion, threshold: float, device, out_dir: Path) -> dict:
    _, probs, labels = predict(model, splits.test_loader, criterion, device)
    predictions = (probs >= threshold).astype(int)

    metrics = {
        "threshold": threshold,
        "accuracy": accuracy_score(labels, predictions),
        "precision": precision_score(labels, predictions, zero_division=0),
        "recall": recall_score(labels, predictions, zero_division=0),
        "f1": f1_score(labels, predictions, zero_division=0),
        "roc_auc": roc_auc_score(labels, probs),
    }

    print("\n" + "=" * 55)
    print("       FINAL TEST SET EVALUATION")
    print("=" * 55)
    print(f"Threshold : {threshold:.3f}")
    print(f"Accuracy  : {metrics['accuracy'] * 100:.2f}%")
    print(f"Precision : {metrics['precision'] * 100:.2f}%")
    print(f"Recall    : {metrics['recall'] * 100:.2f}%")
    print(f"F1-Score  : {metrics['f1']:.4f}")
    print(f"ROC-AUC   : {metrics['roc_auc']:.4f}")
    print("=" * 55)
    print("\n" + classification_report(
        labels, predictions, labels=[0, 1],
        target_names=["Routine Discharge", "High-Risk Discharge"], zero_division=0,
    ))
    print("Confusion matrix:\n", confusion_matrix(labels, predictions, labels=[0, 1]))

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import seaborn as sns

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        sns.heatmap(
            confusion_matrix(labels, predictions, labels=[0, 1]),
            annot=True, fmt="d", cmap="Blues", ax=axes[0],
            xticklabels=["Routine", "High-Risk"],
            yticklabels=["Routine", "High-Risk"],
        )
        axes[0].set_title("Confusion Matrix")
        axes[0].set_xlabel("Predicted")
        axes[0].set_ylabel("Actual")

        fpr, tpr, _ = roc_curve(labels, probs)
        axes[1].plot(fpr, tpr, lw=2, label=f"Fusion (AUC = {metrics['roc_auc']:.4f})")
        axes[1].plot([0, 1], [0, 1], linestyle="--", lw=2, label="Chance (0.50)")
        axes[1].set_xlabel("False Positive Rate")
        axes[1].set_ylabel("True Positive Rate")
        axes[1].set_title("ROC Curve")
        axes[1].legend(loc="lower right")
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        plot_path = out_dir / "multimodal_evaluation_plots.png"
        plt.savefig(plot_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"Evaluation plot -> {plot_path}")
    except Exception as exc:  # noqa: BLE001 - plotting is optional
        print(f"Skipping plots: {exc}")

    metrics_path = out_dir / "test_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"Metrics -> {metrics_path}")

    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--models-dir", type=Path, default=DEFAULT_MODELS_DIR)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--warmup-epochs", type=int, default=TrainConfig.warmup_epochs)
    parser.add_argument("--finetune-epochs", type=int, default=TrainConfig.finetune_epochs)
    parser.add_argument("--batch-size", type=int, default=TrainConfig.batch_size)
    parser.add_argument("--seed", type=int, default=TrainConfig.seed)
    args = parser.parse_args()

    config = TrainConfig(
        warmup_epochs=args.warmup_epochs,
        finetune_epochs=args.finetune_epochs,
        batch_size=args.batch_size,
        seed=args.seed,
    )
    set_seed(config.seed)
    device = resolve_device(args.device)
    print(f"Device: {device}")

    prepared = prepare_datasets(args.data, seed=config.seed)
    print(f"Train samples: {len(prepared.y_train)} | Test samples: {len(prepared.y_test)}")
    print(f"Tabular feature dimension: {prepared.X_train.shape[1]}")
    print(f"Class balance (train): {np.bincount(prepared.y_train)}")

    tokenizer, tokenizer_name = load_tokenizer(DEFAULT_TEXT_ENCODER)
    print(f"Tokenizer: {tokenizer_name}")

    args.models_dir.mkdir(parents=True, exist_ok=True)
    trained = train(prepared, tokenizer, tokenizer_name, config, device, args.models_dir)

    evaluate_test(
        trained["model"],
        trained["splits"],
        trained["criterion"],
        threshold=trained["checkpoint"]["decision_threshold"],
        device=device,
        out_dir=args.models_dir.parent,
    )


if __name__ == "__main__":
    main()
