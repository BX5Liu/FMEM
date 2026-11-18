#!/usr/bin/env python3
# -*- coding: utf-8 -*-


from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn as nn


class TemporalAttentionPool(nn.Module):


    def __init__(self, dim: int, attention_dim: int | None = None):
        super().__init__()
        a = int(attention_dim or max(dim // 2, 8))
        self.proj = nn.Linear(dim, a)
        self.score = nn.Linear(a, 1, bias=False)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:

        logits = self.score(torch.tanh(self.proj(x))).squeeze(-1)
        weights = torch.softmax(logits, dim=1)
        pooled = torch.sum(x * weights.unsqueeze(-1), dim=1)
        return pooled, weights


class AdditiveCrossModalHeads(nn.Module):


    def __init__(self, input_dim: int, output_dim: int, heads: int = 4):
        super().__init__()
        if output_dim % heads != 0:
            raise ValueError("output_dim must be divisible by heads")
        self.heads = int(heads)
        self.head_dim = int(output_dim // heads)
        self.q = nn.ModuleList([nn.Linear(input_dim, self.head_dim) for _ in range(heads)])
        self.k = nn.ModuleList([nn.Linear(input_dim, self.head_dim) for _ in range(heads)])
        self.v = nn.ModuleList([nn.Linear(input_dim, self.head_dim) for _ in range(heads)])
        self.out = nn.Linear(output_dim, output_dim)

    def forward(self, c: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        outputs = []
        weights = []
        for q, k, v in zip(self.q, self.k, self.v):
            qi = q(c)
            ki = k(c)
            vi = v(c)
            wi = torch.softmax(torch.tanh(qi + ki), dim=-1)
            outputs.append(wi * vi)
            weights.append(wi)
        z = self.out(torch.cat(outputs, dim=-1))
        return z, torch.stack(weights, dim=1)


class ACMAStyle(nn.Module):


    def __init__(
        self,
        text_dim: int,
        audio_dim: int,
        hidden: int = 64,
        dropout: float = 0.30,
        heads: int = 4,
    ):
        super().__init__()
        if hidden % 2 != 0:
            raise ValueError("hidden must be even for bidirectional LSTMs")
        per_direction = hidden // 2
        self.text_encoder = nn.LSTM(
            text_dim, per_direction, batch_first=True, bidirectional=True
        )
        self.audio_encoder = nn.LSTM(
            audio_dim, per_direction, batch_first=True, bidirectional=True
        )
        self.text_pool = TemporalAttentionPool(hidden)
        self.audio_pool = TemporalAttentionPool(hidden)
        self.text_fc = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Dropout(dropout))
        self.audio_fc = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Dropout(dropout))
        self.cross = AdditiveCrossModalHeads(hidden * 2, hidden, heads=heads)
        self.norm = nn.LayerNorm(hidden)
        self.classifier = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(self, text: torch.Tensor, audio: torch.Tensor):
        text_seq, _ = self.text_encoder(text)
        audio_seq, _ = self.audio_encoder(audio)
        text_ctx, text_w = self.text_pool(text_seq)
        audio_ctx, audio_w = self.audio_pool(audio_seq)
        c = torch.cat([self.text_fc(text_ctx), self.audio_fc(audio_ctx)], dim=-1)
        cross, cross_w = self.cross(c)
        logits = self.classifier(self.norm(cross)).squeeze(-1)
        return logits, {
            "text_topic_weights": text_w,
            "audio_topic_weights": audio_w,
            "cross_feature_weights": cross_w,
        }


class CircularRecurrentInteraction(nn.Module):


    def __init__(self, dim: int):
        super().__init__()
        self.dim = int(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        recurrent = torch.stack(
            [torch.roll(x, shifts=i, dims=-1) for i in range(self.dim)], dim=-2
        )
        return torch.mean(recurrent * x.unsqueeze(-2), dim=-2)


class AMRFPair(nn.Module):


    def __init__(self, dim: int, dropout: float):
        super().__init__()
        self.recur_x = CircularRecurrentInteraction(dim)
        self.recur_y = CircularRecurrentInteraction(dim)
        self.modal_logits = nn.Parameter(torch.zeros(2))
        self.out = nn.Sequential(
            nn.Linear(dim, dim), nn.GELU(), nn.Dropout(dropout), nn.LayerNorm(dim)
        )

    def forward(self, x: torch.Tensor, y: torch.Tensor):
        xr = self.recur_x(x)
        yr = self.recur_y(y)
        weights = torch.softmax(self.modal_logits, dim=0)
        fused = weights[0] * xr + weights[1] * yr
        return self.out(fused), weights


class CANAMRFATStyle(nn.Module):


    def __init__(
        self,
        text_dim: int,
        audio_dim: int,
        hidden: int = 64,
        dropout: float = 0.30,
        heads: int = 4,
    ):
        super().__init__()
        if hidden % heads != 0:
            raise ValueError("hidden must be divisible by heads")
        self.text_projection = nn.Sequential(
            nn.Conv1d(text_dim, hidden, kernel_size=1), nn.GELU(), nn.Dropout(dropout)
        )
        self.audio_projection = nn.Sequential(
            nn.Conv1d(audio_dim, hidden, kernel_size=1), nn.GELU(), nn.Dropout(dropout)
        )
        self.topic_embedding = nn.Parameter(torch.zeros(1, 3, hidden))
        nn.init.normal_(self.topic_embedding, std=0.02)
        self.amrf = AMRFPair(hidden, dropout)
        self.cross_attention = nn.MultiheadAttention(
            hidden, heads, dropout=dropout, batch_first=True
        )
        self.cross_norm = nn.LayerNorm(hidden)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden,
            nhead=heads,
            dim_feedforward=hidden * 2,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.self_attention = nn.TransformerEncoder(encoder_layer, num_layers=1)
        self.pool = TemporalAttentionPool(hidden)
        self.classifier = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    @staticmethod
    def _project(module: nn.Module, x: torch.Tensor) -> torch.Tensor:
        return module(x.transpose(1, 2)).transpose(1, 2)

    def forward(self, text: torch.Tensor, audio: torch.Tensor):
        t = self._project(self.text_projection, text) + self.topic_embedding[:, : text.shape[1]]
        a = self._project(self.audio_projection, audio) + self.topic_embedding[:, : audio.shape[1]]
        at, modal_w = self.amrf(a, t)
        cross, cross_w = self.cross_attention(t, at, at, need_weights=True)
        z = self.cross_norm(t + cross)
        z = self.self_attention(z)
        pooled, topic_w = self.pool(z)
        logits = self.classifier(pooled).squeeze(-1)
        return logits, {
            "adaptive_modal_weights": modal_w,
            "cross_topic_weights": cross_w,
            "final_topic_weights": topic_w,
        }


def build_architecture(
    name: str,
    text_dim: int,
    audio_dim: int,
    hidden: int,
    dropout: float,
    heads: int,
) -> nn.Module:
    if name == "acma_style":
        return ACMAStyle(text_dim, audio_dim, hidden, dropout, heads)
    if name == "canamrf_at_style":
        return CANAMRFATStyle(text_dim, audio_dim, hidden, dropout, heads)
    raise ValueError(f"Unknown architecture: {name}")


def count_trainable_parameters(model: nn.Module) -> int:
    return int(sum(p.numel() for p in model.parameters() if p.requires_grad))
