import numpy as np
import tensorflow as tf


class LSTMHazard:
    """LSTM-based hazard model for multihorizon failure prediction."""

    def __init__(self, input_shape, n_horizons, config=None):
        c = config or {}
        self.n_horizons = n_horizons
        units = c.get("units", [64, 32])
        dropout = c.get("dropout", 0.2)
        rec_dropout = c.get("recurrent_dropout", 0.2)
        lr = c.get("lr", 0.001)
        self.epochs = c.get("epochs", 100)
        self.batch_size = c.get("batch_size", 32)
        self.patience = c.get("patience", 15)
        self.seed = tf.random.set_seed(42)

        inp = tf.keras.layers.Input(shape=input_shape, name="sequence")
        x = inp
        for i, u in enumerate(units):
            return_seq = i < len(units) - 1
            x = tf.keras.layers.LSTM(
                u, dropout=dropout, recurrent_dropout=rec_dropout,
                return_sequences=return_seq, name=f"lstm_{i}")(x)

        out = tf.keras.layers.Dense(
            n_horizons, activation="sigmoid", name="hazard")(x)

        self.model = tf.keras.Model(inp, out)
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(lr),
            loss="binary_crossentropy",
            metrics=["auc", tf.keras.metrics.BinaryCrossentropy()])

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        callbacks = [tf.keras.callbacks.EarlyStopping(
            monitor="val_loss" if X_val is not None else "loss",
            patience=self.patience, restore_best_weights=True,
            verbose=0)]
        self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val) if X_val is not None else None,
            epochs=self.epochs, batch_size=self.batch_size,
            callbacks=callbacks, verbose=0)
        return self

    def predict_proba(self, X):
        return self.model.predict(X, verbose=0)

    @property
    def name(self):
        return "LSTM"
