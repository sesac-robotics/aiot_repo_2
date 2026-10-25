"""연구 코드에서 가져온 자세 분류기 호환 구현입니다. 저장 모델의 구조·특징 계약을 유지하며 변경 시 원본 연구 코드와의 동등성 검증이 필요합니다.

Temporal multi-label classifiers; sigmoid outputs independent error probabilities.
"""

import torch
from torch import nn


class CNN(nn.Module):
    def __init__(self, features=69, outputs=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(features, 48, 5, padding=2),
            nn.ReLU(),
            nn.Conv1d(48, 64, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Dropout(0.2),
            nn.Linear(64, outputs),
        )

    def forward(self, x):
        return self.net(x.transpose(1, 2))


class GRU(nn.Module):
    def __init__(self, features=69, outputs=2):
        super().__init__()
        self.gru = nn.GRU(features, 48, batch_first=True)
        self.head = nn.Sequential(nn.Dropout(0.2), nn.Linear(48, outputs))

    def forward(self, x):
        out, _ = self.gru(x)
        return self.head(out.mean(dim=1))


MODELS = {"cnn": CNN, "gru": GRU}

from torch import nn


class Hybrid(nn.Module):
    def __init__(self, d, s, n, kind):
        super().__init__()
        self.kind = kind
        if kind == "cnn":
            self.temporal = nn.Sequential(
                nn.Conv1d(d, 48, 5, padding=2),
                nn.ReLU(),
                nn.Conv1d(48, 64, 3, padding=1),
                nn.ReLU(),
                nn.AdaptiveAvgPool1d(1),
                nn.Flatten(),
            )
            temporal_dim = 64
        else:
            self.temporal = nn.GRU(d, 48, batch_first=True, bidirectional=True)
            temporal_dim = 96
        self.stats = nn.Sequential(nn.Linear(s, 96), nn.ReLU(), nn.Dropout(0.3))
        self.head = nn.Sequential(
            nn.Linear(temporal_dim + 96, 48),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(48, n),
        )

    def forward(self, x, s):
        if self.kind == "cnn":
            t = self.temporal(x.transpose(1, 2))
        else:
            t = self.temporal(x)[0].mean(dim=1)
        return self.head(torch.cat([t, self.stats(s)], dim=1))
