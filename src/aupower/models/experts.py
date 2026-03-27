from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.preprocessing import StandardScaler

try:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset
except ImportError as exc:  # pragma: no cover
    raise RuntimeError("torch is required to use aupower.models.experts") from exc

from aupower.config import ForecastConfig
from aupower.data.dataset import SampleBundle


def _fit_scaler(values: np.ndarray) -> StandardScaler | None:
    if values.size == 0 or values.shape[1] == 0:
        return None
    scaler = StandardScaler()
    scaler.fit(values)
    return scaler


def _apply_scaler(values: np.ndarray, scaler: StandardScaler | None) -> np.ndarray:
    if scaler is None or values.size == 0 or values.shape[1] == 0:
        return values.astype(np.float32)
    return scaler.transform(values).astype(np.float32)


class ExpertNet(nn.Module):
    def __init__(self, load_dim: int, calendar_dim: int, weather_dim: int, event_dim: int, hidden_dim: int, horizon: int, dropout: float):
        super().__init__()
        self.load_encoder = nn.Sequential(
            nn.Linear(load_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        self.calendar_encoder = nn.Sequential(
            nn.Linear(calendar_dim, hidden_dim // 2),
            nn.GELU(),
        )
        self.weather_encoder = (
            nn.Sequential(nn.Linear(weather_dim, hidden_dim // 4), nn.GELU()) if weather_dim > 0 else None
        )
        self.event_encoder = (
            nn.Sequential(nn.Linear(event_dim, hidden_dim // 4), nn.GELU()) if event_dim > 0 else None
        )
        fusion_dim = hidden_dim + hidden_dim // 2
        fusion_dim += hidden_dim // 4 if weather_dim > 0 else 0
        fusion_dim += hidden_dim // 4 if event_dim > 0 else 0
        self.fusion = nn.Sequential(
            nn.Linear(fusion_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, horizon),
        )

    def forward(self, load_x, calendar_x, weather_x=None, event_x=None):
        blocks = [self.load_encoder(load_x), self.calendar_encoder(calendar_x)]
        if self.weather_encoder is not None and weather_x is not None and weather_x.shape[1] > 0:
            blocks.append(self.weather_encoder(weather_x))
        if self.event_encoder is not None and event_x is not None and event_x.shape[1] > 0:
            blocks.append(self.event_encoder(event_x))
        return self.fusion(torch.cat(blocks, dim=1))


@dataclass
class WindowExpert:
    name: str
    config: ForecastConfig
    model: ExpertNet | None = None
    load_scaler: StandardScaler | None = None
    calendar_scaler: StandardScaler | None = None
    weather_scaler: StandardScaler | None = None
    event_scaler: StandardScaler | None = None
    target_scaler: StandardScaler | None = None
    metadata: dict[str, Any] | None = None

    def _build_model(self, samples: SampleBundle) -> None:
        self.metadata = {
            "load_dim": samples.load_history.shape[1],
            "calendar_dim": samples.calendar_future.shape[1],
            "weather_dim": samples.weather_future.shape[1] if samples.weather_future.ndim == 2 else 0,
            "event_dim": samples.event_context.shape[1] if samples.event_context.ndim == 2 else 0,
            "horizon": self.config.experts.horizon,
            "hidden_dim": self.config.experts.hidden_dim,
            "dropout": self.config.experts.dropout,
        }
        self.model = ExpertNet(**self.metadata)

    def _scale_samples(self, samples: SampleBundle) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        load_x = _apply_scaler(samples.load_history, self.load_scaler)
        calendar_x = _apply_scaler(samples.calendar_future, self.calendar_scaler)
        weather_x = _apply_scaler(samples.weather_future, self.weather_scaler)
        event_x = _apply_scaler(samples.event_context, self.event_scaler)
        target_y = _apply_scaler(samples.target, self.target_scaler)
        return load_x, calendar_x, weather_x, event_x, target_y

    def initialize_from(self, other: "WindowExpert") -> None:
        if self.model is None or other.model is None:
            return
        own_state = self.model.state_dict()
        shared = {key: value for key, value in other.model.state_dict().items() if key in own_state and own_state[key].shape == value.shape}
        own_state.update(shared)
        self.model.load_state_dict(own_state)

    def fit(self, train: SampleBundle, valid: SampleBundle, init_from: "WindowExpert" | None = None) -> "WindowExpert":
        if train.target.size == 0:
            raise ValueError(f"{self.name} received no training samples")
        self._build_model(train)
        if init_from is not None:
            self.initialize_from(init_from)

        self.load_scaler = _fit_scaler(train.load_history)
        self.calendar_scaler = _fit_scaler(train.calendar_future)
        self.weather_scaler = _fit_scaler(train.weather_future)
        self.event_scaler = _fit_scaler(train.event_context)
        self.target_scaler = _fit_scaler(train.target)

        train_arrays = self._scale_samples(train)
        valid_arrays = self._scale_samples(valid)

        train_dataset = TensorDataset(*[torch.from_numpy(array) for array in train_arrays])
        valid_dataset = TensorDataset(*[torch.from_numpy(array) for array in valid_arrays])
        train_loader = DataLoader(train_dataset, batch_size=self.config.experts.batch_size, shuffle=True)
        valid_loader = DataLoader(valid_dataset, batch_size=self.config.experts.batch_size, shuffle=False)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config.experts.lr)
        criterion = nn.MSELoss()
        best_state = None
        best_loss = float("inf")
        patience_left = self.config.experts.patience
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(device)

        for _epoch in range(self.config.experts.max_epochs):
            self.model.train()
            for batch in train_loader:
                load_x, cal_x, weather_x, event_x, target_y = [tensor.to(device) for tensor in batch]
                optimizer.zero_grad()
                pred = self.model(load_x, cal_x, weather_x, event_x)
                loss = criterion(pred, target_y)
                loss.backward()
                optimizer.step()

            self.model.eval()
            losses = []
            with torch.no_grad():
                for batch in valid_loader:
                    load_x, cal_x, weather_x, event_x, target_y = [tensor.to(device) for tensor in batch]
                    pred = self.model(load_x, cal_x, weather_x, event_x)
                    losses.append(float(criterion(pred, target_y).detach().cpu().item()))
            valid_loss = float(np.mean(losses)) if losses else float("inf")
            if valid_loss < best_loss:
                best_loss = valid_loss
                best_state = {key: value.detach().cpu() for key, value in self.model.state_dict().items()}
                patience_left = self.config.experts.patience
            else:
                patience_left -= 1
                if patience_left <= 0:
                    break

        if best_state is not None:
            self.model.load_state_dict(best_state)
        return self

    def predict(self, samples: SampleBundle) -> np.ndarray:
        if self.model is None:
            raise RuntimeError(f"{self.name} must be fitted or loaded before calling predict()")
        load_x, calendar_x, weather_x, event_x, _ = self._scale_samples(samples)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(device)
        self.model.eval()
        with torch.no_grad():
            predictions = self.model(
                torch.from_numpy(load_x).to(device),
                torch.from_numpy(calendar_x).to(device),
                torch.from_numpy(weather_x).to(device),
                torch.from_numpy(event_x).to(device),
            ).detach().cpu().numpy()
        if self.target_scaler is not None:
            predictions = self.target_scaler.inverse_transform(predictions)
        return predictions.astype(np.float32)

    def save(self, path: str | Path) -> None:
        if self.model is None or self.metadata is None:
            raise RuntimeError("cannot save an unfitted expert")
        payload = {
            "name": self.name,
            "metadata": self.metadata,
            "state_dict": self.model.state_dict(),
            "load_scaler": self.load_scaler,
            "calendar_scaler": self.calendar_scaler,
            "weather_scaler": self.weather_scaler,
            "event_scaler": self.event_scaler,
            "target_scaler": self.target_scaler,
            "config": self.config.to_dict(),
        }
        torch.save(payload, Path(path))

    @classmethod
    def load(cls, path: str | Path, config: ForecastConfig) -> "WindowExpert":
        payload = torch.load(Path(path), map_location="cpu")
        expert = cls(name=payload["name"], config=config)
        expert.metadata = payload["metadata"]
        expert.model = ExpertNet(**expert.metadata)
        expert.model.load_state_dict(payload["state_dict"])
        expert.load_scaler = payload["load_scaler"]
        expert.calendar_scaler = payload["calendar_scaler"]
        expert.weather_scaler = payload["weather_scaler"]
        expert.event_scaler = payload["event_scaler"]
        expert.target_scaler = payload["target_scaler"]
        return expert
