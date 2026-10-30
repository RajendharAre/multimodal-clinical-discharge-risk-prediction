"""Data loading, preprocessing, and PyTorch dataset construction."""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset, Subset

SEED = 42
MAX_LEN = 128
BATCH_SIZE = 16
VAL_FRACTION = 0.15

HIGH_RISK_TYPES = ("lama", "discharged on request", "shifted to higher center")
VALID_DISCHARGE_TYPES = ("discharged", *HIGH_RISK_TYPES)

TEXT_COLUMNS = ("Chief_Complaints", "Hospital_Course_Treatment", "Primary_Diagnosis")

NUMERIC_COLUMNS = [
    "Age",
    "Systolic_BP_mmHg",
    "Diastolic_BP_mmHg",
    "Heart_Rate_bpm",
    "SpO2_Percent",
    "Body_Temp_F",
    "Respiratory_Rate_bpm",
    "GCS_Score",
    "Est_Out_Of_Pocket_Expense_INR",
    "Prior_Hospitalizations_12M",
    "Active_Prescription_Count",
    "Charlson_Comorbidity_Index",
    "Admission_Creatinine_mg_dL",
    "Admission_Hb_g_dL",
    "Distance_From_Hospital_km",
]

CATEGORICAL_COLUMNS = [
    "Sex",
    "Admission_Type",
    "Triage_Priority",
    "Referral_Source",
    "Admission_Shift",
    "Insurance_Type",
    "Primary_Caregiver_Available",
]

_AGE_RANGE_RE = re.compile(r"^(\d+(?:\.\d+)?)\s*-\s*(\d+(?:\.\d+)?)$")
_NUMBER_RE = re.compile(r"\d+(?:\.\d+)?")

# Dataset encodes the open brackets as text; these are representative
# midpoints so paediatric, elderly, and remote patients keep a distinct signal.
_AGE_OPEN_LOW = 10.0
_AGE_OPEN_HIGH = 80.0
_DISTANCE_OPEN_HIGH = 65.0


def _clean_text(value: object) -> str | None:
    if value is None:
        return None
    if isinstance(value, float) and np.isnan(value):
        return None
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none"}:
        return None
    return text


def _parse_bracket(text: str, open_low: float, open_high: float) -> float:
    if text.startswith("<"):
        return open_low
    if text.endswith("+"):
        return open_high

    match = _AGE_RANGE_RE.match(text)
    if match:
        low, high = match.groups()
        return (float(low) + float(high)) / 2.0

    try:
        return float(text)
    except ValueError:
        return np.nan


def parse_age(value: object) -> float:
    text = _clean_text(value)
    if text is None:
        return np.nan
    return _parse_bracket(text, _AGE_OPEN_LOW, _AGE_OPEN_HIGH)


def parse_distance(value: object) -> float:
    text = _clean_text(value)
    if text is None:
        return np.nan
    text = text.lower().replace("km", "").strip()
    if text.endswith("+"):
        return _DISTANCE_OPEN_HIGH

    numbers = _NUMBER_RE.findall(text)
    if len(numbers) == 2:
        return (float(numbers[0]) + float(numbers[1])) / 2.0
    if len(numbers) == 1:
        return float(numbers[0])
    return np.nan


@dataclass
class Preprocessor:
    """Fitted statistics that convert a raw patient frame to a tabular matrix.

    Persisted as plain JSON so a preprocessing artifact stays valid across
    scikit-learn versions; fitting and inference never unpickle library state.
    """

    medians: list[float]
    means: list[float]
    scales: list[float]
    categories: list[list[str]]
    numeric_cols: list[str]
    categorical_cols: list[str]

    @classmethod
    def fit(cls, frame: pd.DataFrame) -> "Preprocessor":
        numeric = frame[NUMERIC_COLUMNS].apply(pd.to_numeric, errors="coerce")
        medians = numeric.median()
        filled = numeric.fillna(medians)
        means = filled.mean()
        scales = filled.std(ddof=0).replace(0.0, 1.0)

        categories = []
        for column in CATEGORICAL_COLUMNS:
            values = frame[column].astype("string").fillna("Missing").astype(str)
            categories.append(sorted(values.unique().tolist()))

        return cls(
            medians=[float(value) for value in medians],
            means=[float(value) for value in means],
            scales=[float(value) for value in scales],
            categories=categories,
            numeric_cols=list(NUMERIC_COLUMNS),
            categorical_cols=list(CATEGORICAL_COLUMNS),
        )

    def transform(self, frame: pd.DataFrame) -> np.ndarray:
        numeric = (
            frame[self.numeric_cols]
            .apply(pd.to_numeric, errors="coerce")
            .to_numpy(dtype=np.float64)
        )
        numeric = np.where(np.isnan(numeric), np.asarray(self.medians), numeric)
        numeric = (numeric - np.asarray(self.means)) / np.asarray(self.scales)

        onehot = np.zeros(
            (len(frame), sum(len(cats) for cats in self.categories)), dtype=np.float32
        )
        offset = 0
        for column, cats in zip(self.categorical_cols, self.categories):
            values = (
                frame[column].astype("string").fillna("Missing").astype(str).to_numpy()
            )
            for index, category in enumerate(cats):
                onehot[:, offset + index] = values == category
            offset += len(cats)

        return np.concatenate([numeric, onehot], axis=1).astype(np.float32)

    @property
    def feature_names(self) -> list[str]:
        names = list(self.numeric_cols)
        for column, cats in zip(self.categorical_cols, self.categories):
            names.extend(f"{column}_{category}" for category in cats)
        return names

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "Preprocessor":
        return cls(**json.loads(Path(path).read_text(encoding="utf-8")))


@dataclass
class PreparedData:
    train_frame: pd.DataFrame
    test_frame: pd.DataFrame
    X_train: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray
    preprocessor: Preprocessor

    @property
    def feature_names(self) -> list[str]:
        return self.preprocessor.feature_names


def build_clinical_text(frame: pd.DataFrame) -> pd.Series:
    text = (
        "Chief Complaints: "
        + frame["Chief_Complaints"].fillna("Not specified").astype(str)
        + " | Hospital Course: "
        + frame["Hospital_Course_Treatment"].fillna("Not specified").astype(str)
        + " | Diagnosis: "
        + frame["Primary_Diagnosis"].fillna("Not specified").astype(str)
    )
    return text


def prepare_datasets(
    csv_path: str | Path,
    test_size: float = 0.2,
    seed: int = SEED,
) -> PreparedData:
    frame = pd.read_csv(csv_path)

    discharge_type = (
        frame["Discharge_Type"].astype("string").str.strip().str.lower()
    )
    frame = frame.loc[discharge_type.isin(VALID_DISCHARGE_TYPES)].copy()
    discharge_type = discharge_type.loc[frame.index]

    frame["High_Risk_Discharge"] = (
        discharge_type.isin(HIGH_RISK_TYPES).astype(int)
    )
    frame["Age"] = frame["Age"].apply(parse_age)
    frame["Distance_From_Hospital_km"] = frame["Distance_From_Hospital_km"].apply(
        parse_distance
    )
    frame["clinical_text"] = build_clinical_text(frame)

    for column in NUMERIC_COLUMNS:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")

    unparsed = [column for column in NUMERIC_COLUMNS if frame[column].isna().all()]
    if unparsed:
        raise ValueError(f"Numeric columns parsed to all-missing: {unparsed}")

    frame = frame.loc[frame[NUMERIC_COLUMNS].notna().any(axis=1)].copy()

    if frame["High_Risk_Discharge"].nunique() != 2:
        raise ValueError("Both target classes must be present.")

    train_frame, test_frame = train_test_split(
        frame,
        test_size=test_size,
        random_state=seed,
        stratify=frame["High_Risk_Discharge"],
    )
    train_frame = train_frame.copy()
    test_frame = test_frame.copy()

    preprocessor = Preprocessor.fit(train_frame)

    return PreparedData(
        train_frame=train_frame,
        test_frame=test_frame,
        X_train=preprocessor.transform(train_frame),
        X_test=preprocessor.transform(test_frame),
        y_train=train_frame["High_Risk_Discharge"].to_numpy(dtype=np.int64),
        y_test=test_frame["High_Risk_Discharge"].to_numpy(dtype=np.int64),
        preprocessor=preprocessor,
    )


class MultiModalDataset(Dataset):
    def __init__(
        self,
        texts: list[str],
        tabular_features: np.ndarray,
        labels: np.ndarray,
        tokenizer,
        max_len: int = MAX_LEN,
    ) -> None:
        self.texts = texts
        self.tabular = torch.tensor(tabular_features, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.float32)
        self.tokenizer = tokenizer
        self.max_len = max_len

        if not len(self.texts) == len(self.tabular) == len(self.labels):
            raise ValueError("Texts, tabular features, and labels must align.")

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        encoding = self.tokenizer(
            str(self.texts[index]),
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt",
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "tabular": self.tabular[index],
            "label": self.labels[index],
        }


@dataclass
class TrainingSplits:
    train_loader: DataLoader
    val_loader: DataLoader
    test_loader: DataLoader
    pos_weight: torch.Tensor
    train_labels: np.ndarray
    val_labels: np.ndarray


def make_loaders(
    prepared: PreparedData,
    tokenizer,
    batch_size: int = BATCH_SIZE,
    max_len: int = MAX_LEN,
    val_fraction: float = VAL_FRACTION,
    seed: int = SEED,
) -> TrainingSplits:
    dataset = MultiModalDataset(
        texts=prepared.train_frame["clinical_text"].tolist(),
        tabular_features=prepared.X_train,
        labels=prepared.y_train,
        tokenizer=tokenizer,
        max_len=max_len,
    )
    test_dataset = MultiModalDataset(
        texts=prepared.test_frame["clinical_text"].tolist(),
        tabular_features=prepared.X_test,
        labels=prepared.y_test,
        tokenizer=tokenizer,
        max_len=max_len,
    )

    train_indices, val_indices = train_test_split(
        np.arange(len(dataset)),
        test_size=val_fraction,
        random_state=seed,
        stratify=prepared.y_train,
    )

    train_labels = prepared.y_train[train_indices]
    val_labels = prepared.y_train[val_indices]

    num_neg = int((train_labels == 0).sum())
    num_pos = int((train_labels == 1).sum())
    if num_neg == 0 or num_pos == 0:
        raise ValueError("Training subset must contain both classes.")

    return TrainingSplits(
        train_loader=DataLoader(
            Subset(dataset, train_indices),
            batch_size=batch_size,
            shuffle=True,
        ),
        val_loader=DataLoader(
            Subset(dataset, val_indices),
            batch_size=batch_size,
            shuffle=False,
        ),
        test_loader=DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
        ),
        pos_weight=torch.tensor(num_neg / num_pos, dtype=torch.float32),
        train_labels=train_labels,
        val_labels=val_labels,
    )
