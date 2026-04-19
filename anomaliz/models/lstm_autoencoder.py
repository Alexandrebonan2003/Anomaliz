from __future__ import annotations

import json
from pathlib import Path

import numpy as np


class LSTMAutoencoder:
    def __init__(
        self,
        units_1: int = 64,
        units_2: int = 32,
        dropout: float = 0.2,
        learning_rate: float = 1e-3,
        epochs: int = 20,
        batch_size: int = 32,
        patience: int = 5,
        random_state: int = 42,
    ) -> None:
        self.params = {
            "units_1": units_1,
            "units_2": units_2,
            "dropout": dropout,
            "learning_rate": learning_rate,
            "epochs": epochs,
            "batch_size": batch_size,
            "patience": patience,
            "random_state": random_state,
        }
        self._model = None
        self._window_size: int | None = None
        self._n_features: int | None = None
        self._err_min: float | None = None
        self._err_max: float | None = None

    def _build(self, window_size: int, n_features: int):
        import tensorflow as tf
        from tensorflow import keras
        from tensorflow.keras import layers

        tf.random.set_seed(self.params["random_state"])

        inp = keras.Input(shape=(window_size, n_features))
        x = layers.LSTM(self.params["units_1"], return_sequences=True)(inp)
        x = layers.Dropout(self.params["dropout"])(x)
        x = layers.LSTM(self.params["units_2"], return_sequences=False)(x)
        x = layers.RepeatVector(window_size)(x)
        x = layers.LSTM(self.params["units_2"], return_sequences=True)(x)
        x = layers.Dropout(self.params["dropout"])(x)
        x = layers.LSTM(self.params["units_1"], return_sequences=True)(x)
        out = layers.TimeDistributed(layers.Dense(n_features))(x)

        model = keras.Model(inp, out)
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=self.params["learning_rate"]),
            loss="mse",
        )
        return model

    def fit(self, X: np.ndarray) -> "LSTMAutoencoder":
        from tensorflow import keras

        X = np.asarray(X, dtype=np.float32)
        if X.ndim != 3:
            raise ValueError("LSTMAutoencoder expects 3D windows (batch, timesteps, features)")
        self._window_size = int(X.shape[1])
        self._n_features = int(X.shape[2])
        self._model = self._build(self._window_size, self._n_features)

        callbacks = []
        if self.params["patience"] > 0:
            callbacks.append(
                keras.callbacks.EarlyStopping(
                    monitor="loss",
                    patience=self.params["patience"],
                    restore_best_weights=True,
                )
            )

        self._model.fit(
            X,
            X,
            epochs=self.params["epochs"],
            batch_size=self.params["batch_size"],
            shuffle=True,
            verbose=0,
            callbacks=callbacks,
        )

        errors = self._reconstruction_error(X)
        self._err_min = float(errors.min())
        self._err_max = float(errors.max())
        return self

    def reconstruction_error(self, X: np.ndarray) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("Detector must be fit before reconstruction_error")
        X = np.asarray(X, dtype=np.float32)
        return self._reconstruction_error(X)

    def _reconstruction_error(self, X: np.ndarray) -> np.ndarray:
        recon = self._model.predict(X, verbose=0)
        return np.mean(np.abs(X - recon), axis=(1, 2))

    def set_error_range(self, err_min: float, err_max: float) -> None:
        self._err_min = float(err_min)
        self._err_max = float(err_max)

    def score(self, X: np.ndarray) -> np.ndarray:
        if self._model is None or self._err_min is None or self._err_max is None:
            raise RuntimeError("Detector must be fit before scoring")
        errors = self._reconstruction_error(np.asarray(X, dtype=np.float32))
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
    def load(cls, path: Path) -> "LSTMAutoencoder":
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
