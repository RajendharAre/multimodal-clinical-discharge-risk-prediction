"""Multimodal clinical text + tabular fusion model."""

from __future__ import annotations

import torch
import torch.nn as nn
from transformers import AutoModel

DEFAULT_TEXT_ENCODER = "emilyalsentzer/Bio_ClinicalBERT"


class BioClinicalMultiModalFusion(nn.Module):
    def __init__(
        self,
        num_tabular_features: int,
        bert_model_name: str = DEFAULT_TEXT_ENCODER,
        hidden_dim: int = 64,
        dropout_rate: float = 0.3,
    ) -> None:
        super().__init__()

        self.bert = AutoModel.from_pretrained(bert_model_name)
        text_dim = self.bert.config.hidden_size

        self.tabular_encoder = nn.Sequential(
            nn.Linear(num_tabular_features, hidden_dim * 2),
            nn.BatchNorm1d(hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
        )

        self.classifier = nn.Sequential(
            nn.Linear(text_dim + hidden_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        tabular: torch.Tensor,
    ) -> torch.Tensor:
        bert_output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        text_embedding = bert_output.last_hidden_state[:, 0, :]
        tabular_embedding = self.tabular_encoder(tabular)
        fused = torch.cat((text_embedding, tabular_embedding), dim=1)
        return self.classifier(fused).squeeze(1)
