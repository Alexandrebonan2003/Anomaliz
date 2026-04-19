from __future__ import annotations

import json
from pathlib import Path

import numpy as np


class LSTMForecaster:
    """Forecasting-based detector: predict the last timestep of each window
    from the prefix and score anomalies by forecast residual magnitude."""

    def __init__(
        self,
        units: int = 32,
        dropout: float = 0.2,
        recurrent_dropout: float = 0.0,
        learning_rate: float = 1e-3,
        epochs: int = 20,
        batch_size: int = 32,
        patience: int = 5,
        val_split: float = 0.1,
        random_state: int = 42,
    ) -> None:
        self.params = {
            "units": units,
            "dropout": dropout,
            "recurrent_dropout": recurrent_dropout,
            "learning_rate": learning_rate,
            "epochs": epochs,
            "batch_size": batch_size,
            "patience": patience,
            "val_split": val_split,
            "random_state": random_state,
        }
        self._model = None
        self._window_size: int | None = None
        self._n_features: int | None = None
        self._err_min: float | None = None
        self._err_max: float | None = None

    def _build(self, input_len: int, n_features: int):
        import tensorflow as tf
        from tensorflow import keras
        from tensorflow.keras import layers

        tf.random.set_seed(self.params["random_state"])

        inp = keras.Input(shape=(input_len, n_features))
        x = layers.LSTM(
            self.params["units"],
            return_sequences=False,
            dropout=self.params["dropout"],
            recurrent_dropout=self.params["recurrent_dropout"],
        )(inp)
        out = layers.Dense(n_features)(x)

        model = keras.Model(inp, out)
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=self.params["learning_rate"]),
            loss="mse",
        )
        return model

    @staticmethod
    def _split_xy(W: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        W = np.asarray(W, dtype=np.float32)
        if W.ndim != 3:
            raise ValueError("LSTMForecaster expects 3D windows (batch, timesteps, features)")
        if W.shape[1] < 2:
            raise ValueError("LSTMForecaster requires window_size >= 2")
        return W[:, :-1, :], W[:, -1, :]

    def fit(self, W: np.ndarray) -> "LSTMForecaster":
        from tensorflow import keras

        X, y = self._split_xy(W)
        self._window_size = int(W.shape[1])
        self._n_features = int(W.shape[2])
        self._model = self._build(X.shape[1], self._n_features)

        val_split = float(self.params["val_split"])
        use_val = val_split > 0.0 and X.shape[0] >= 4
        monitor = "val_loss" if use_val else "loss"

        callbacks = []
        if self.params["patience"] > 0:
            callbacks.append(
                keras.callbacks.EarlyStopping(
                    monitor=monitor,
                    patience=self.params["patience"],
                    restore_best_weights=True,
                )
            )

        self._model.fit(
            X,
            y,
            epochs=self.params["epochs"],
            batch_size=self.params["batch_size"],
            validation_split=val_split if use_val else 0.0,
            shuffle=True,
            verbose=0,
            callbacks=callbacks,
        )

        errors = self._residuals(W)
        self._err_min = float(errors.min())
        self._err_max = float(errors.max())
        return self

    def forecast_residuals(self, W: np.ndarray) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("Detector must be fit before forecast_residuals")
        return self._residuals(np.asarray(W, dtype=np.float32))

    def _residuals(self, W: np.ndarray) -> np.ndarray:
        X, y = self._split_xy(W)
        pred = self._model.predict(X, verbose=0)
        return np.mean(np.abs(y - pred), axis=1)

    def set_error_range(self, err_min: float, err_max: float) -> None:
        self._err_min = float(err_min)
        self._err_max = float(err_max)

    def score(self, W: np.ndarray) -> np.ndarray:
        if self._model is None or self._err_min is None or self._err_max is None:
            raise RuntimeError("Detector must be fit before scoring")
        errors = self._residuals(np.asarray(W, dtype=np.float32))
        span = self._err_max - self._err_min
        if span <= 0:
            span = 1.0
        return np.clip((errors - self._err_min) / span, 0.0, 1.0)

    def save(self, path: Path) -> None:
        if self._model is None:
            raise RuntimeError("Cannot save an unfit detector")
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        self._model.save(path / "model.keras")
        (path / "stats.json").write_text(
            json.dumps(
                {
                    "err_min": self._err_min,
                    "err_max": self._err_max,
                    "window_size": self._window_size,
                    "n_features": self._n_features,
                    "params": self.params,
                }
            )
        )

    @classmethod
    def load(cls, path: Path) -> "LSTMForecaster":
        from tensorflow import keras

        path = Path(path)
        stats = json.loads((path / "stats.json").read_text())
        inst = cls(**stats["params"])
        inst._model = keras.models.load_model(path / "model.keras")
        inst._err_min = stats["err_min"]
        inst._err_max = stats["err_max"]
        inst._window_size = stats["window_size"]
        inst._n_features = stats["n_features"]
        return inst
