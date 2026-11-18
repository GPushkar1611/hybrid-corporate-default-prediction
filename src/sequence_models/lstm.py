"""Small PyTorch LSTM classifier for fixed-length firm sequences."""

from __future__ import annotations

import numpy as np
import torch
from torch import nn
from sklearn.metrics import average_precision_score


class _LSTMNetwork(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, dropout: float, num_layers: int):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )
        self.dropout = nn.Dropout(dropout)
        self.output = nn.Linear(hidden_size, 1)

    def forward(self, values):
        sequence_output, _ = self.lstm(values)
        final_state = sequence_output[:, -1, :]
        return self.output(self.dropout(final_state)).squeeze(-1)


class LSTMClassifier:
    """Binary sequence classifier with train-derived positive-class weighting."""

    def __init__(
        self,
        *,
        hidden_size: int = 32,
        dropout: float = 0.2,
        learning_rate: float = 1e-3,
        epochs: int = 10,
        batch_size: int = 256,
        random_state: int = 42,
        device: str = "cpu",
        num_layers: int = 2,
        weight_decay: float = 1e-4,
        gradient_clip_norm: float = 1.0,
        patience: int = 5,
        min_delta: float = 1e-4,
    ):
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.random_state = random_state
        self.device = torch.device(device)
        self.num_layers = num_layers
        self.weight_decay = weight_decay
        self.gradient_clip_norm = gradient_clip_norm
        self.patience = patience
        self.min_delta = min_delta
        self.network: _LSTMNetwork | None = None
        self.training_history: list[float] = []
        self.validation_average_precision: list[float] = []
        self.best_epoch: int | None = None
        self.best_validation_average_precision: float | None = None
        self.scale_pos_weight: float | None = None

    def fit(self, X, y, *, validation_data=None) -> "LSTMClassifier":
        values = np.asarray(X, dtype=np.float32)
        target = np.asarray(y, dtype=np.int64)
        if values.ndim != 3:
            raise ValueError("X must have shape (samples, time steps, features)")
        if len(values) != len(target):
            raise ValueError("X and y must have the same sample count")
        if not np.isfinite(values).all():
            raise ValueError("LSTM input must be finite; impute and scale using training rows first")
        if set(np.unique(target)) != {0, 1}:
            raise ValueError("Training labels must contain both binary classes")
        if self.epochs < 1 or self.batch_size < 1:
            raise ValueError("epochs and batch_size must be positive")

        torch.manual_seed(self.random_state)
        if torch.cuda.is_available() and self.device.type == "cuda":
            torch.cuda.manual_seed_all(self.random_state)
        torch.set_num_threads(max(1, min(torch.get_num_threads(), 4)))

        if self.num_layers < 1 or self.patience < 1 or self.weight_decay < 0:
            raise ValueError("num_layers and patience must be positive; weight_decay cannot be negative")
        self.network = _LSTMNetwork(
            values.shape[2], self.hidden_size, self.dropout, self.num_layers
        ).to(self.device)
        x_tensor = torch.from_numpy(np.ascontiguousarray(values)).to(self.device)
        y_tensor = torch.from_numpy(target.astype(np.float32, copy=False)).to(self.device)
        positives = float(target.sum())
        negatives = float(len(target) - target.sum())
        self.scale_pos_weight = negatives / positives
        loss_fn = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor([self.scale_pos_weight], dtype=torch.float32, device=self.device)
        )
        optimizer = torch.optim.AdamW(
            self.network.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay
        )
        generator = torch.Generator(device="cpu").manual_seed(self.random_state)
        self.training_history = []
        self.validation_average_precision = []
        validation_X = validation_y = None
        if validation_data is not None:
            validation_X = np.asarray(validation_data[0], dtype=np.float32)
            validation_y = np.asarray(validation_data[1], dtype=np.int64)
            if validation_X.ndim != 3 or validation_X.shape[2] != values.shape[2]:
                raise ValueError("Validation X must have the same feature count as training X")
            if len(validation_X) != len(validation_y) or not np.isfinite(validation_X).all():
                raise ValueError("Validation data must have matching samples and finite features")
            if set(np.unique(validation_y)) != {0, 1}:
                raise ValueError("Validation labels must contain both binary classes")
            validation_tensor = torch.from_numpy(np.ascontiguousarray(validation_X)).to(self.device)
        best_state = None
        best_score = -np.inf
        epochs_without_improvement = 0

        for epoch in range(self.epochs):
            self.network.train()
            order = torch.randperm(len(target), generator=generator)
            weighted_loss = 0.0
            for start in range(0, len(target), self.batch_size):
                indices = order[start : start + self.batch_size].to(self.device)
                optimizer.zero_grad(set_to_none=True)
                logits = self.network(x_tensor[indices])
                loss = loss_fn(logits, y_tensor[indices])
                loss.backward()
                if self.gradient_clip_norm > 0:
                    nn.utils.clip_grad_norm_(self.network.parameters(), self.gradient_clip_norm)
                optimizer.step()
                weighted_loss += float(loss.detach().cpu()) * len(indices)
            self.training_history.append(weighted_loss / len(target))
            if validation_data is not None:
                self.network.eval()
                with torch.no_grad():
                    logits = self.network(validation_tensor)
                    probabilities = torch.sigmoid(logits).cpu().numpy()
                score = float(average_precision_score(validation_y, probabilities))
                self.validation_average_precision.append(score)
                if score > best_score + self.min_delta:
                    best_score = score
                    self.best_epoch = epoch + 1
                    best_state = {key: value.detach().cpu().clone() for key, value in self.network.state_dict().items()}
                    epochs_without_improvement = 0
                else:
                    epochs_without_improvement += 1
                    if epochs_without_improvement >= self.patience:
                        break
        if best_state is not None:
            self.network.load_state_dict(best_state)
            self.best_validation_average_precision = float(best_score)
        return self

    def predict_proba(self, X) -> np.ndarray:
        if self.network is None:
            raise RuntimeError("Fit the LSTM before prediction")
        values = np.asarray(X, dtype=np.float32)
        if values.ndim != 3 or values.shape[2] != self.network.lstm.input_size:
            raise ValueError("X must have shape (samples, time steps, fitted feature count)")
        if not np.isfinite(values).all():
            raise ValueError("LSTM input must be finite; impute and scale using training rows first")
        tensor = torch.from_numpy(np.ascontiguousarray(values)).to(self.device)
        self.network.eval()
        probabilities = []
        with torch.no_grad():
            for start in range(0, len(tensor), self.batch_size):
                logits = self.network(tensor[start : start + self.batch_size])
                probabilities.append(torch.sigmoid(logits).cpu().numpy())
        return np.concatenate(probabilities) if probabilities else np.empty(0, dtype=float)
