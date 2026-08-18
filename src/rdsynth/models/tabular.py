from __future__ import annotations

import torch
from torch import nn


class TabularCNN(nn.Module):
    def __init__(self, in_dim: int, channels: int = 32, out_dim: int = 2):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(1, channels, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv1d(channels, channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.fc = nn.Linear(channels, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.unsqueeze(1)
        h = self.conv(x).squeeze(-1)
        return self.fc(h)


class TabularRNN(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int = 2):
        super().__init__()
        self.rnn = nn.RNN(input_size=1, hidden_size=hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq = x.unsqueeze(-1)
        _, h = self.rnn(seq)
        return self.fc(h[-1])


class TabularLSTM(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int = 2):
        super().__init__()
        self.lstm = nn.LSTM(input_size=1, hidden_size=hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq = x.unsqueeze(-1)
        _, (h, _) = self.lstm(seq)
        return self.fc(h[-1])


class TabularGRU(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int = 2):
        super().__init__()
        self.gru = nn.GRU(input_size=1, hidden_size=hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq = x.unsqueeze(-1)
        _, h = self.gru(seq)
        return self.fc(h[-1])


class TabularTransformer(nn.Module):
    def __init__(self, in_dim: int, d_model: int = 64, nhead: int = 4, num_layers: int = 2, out_dim: int = 2):
        super().__init__()
        self.input_proj = nn.Linear(1, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(d_model, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq = x.unsqueeze(-1)
        h = self.input_proj(seq)
        h = self.encoder(h)
        pooled = h.mean(dim=1)
        return self.fc(pooled)
