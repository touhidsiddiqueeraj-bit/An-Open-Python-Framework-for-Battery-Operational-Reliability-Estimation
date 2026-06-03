import numpy as np
import tensorflow as tf
from tensorflow.keras import layers


class TCNHazard:
    """Temporal Convolutional Network for multihorizon hazard prediction.

    Uses dilated causal convolutions to capture long-range dependencies
    in degradation sequences.
    """

    def __init__(self, input_shape, n_horizons, config=None):
        c = config or {}
        filters = c.get("filters", [32, 32])
        kernel_size = c.get("kernel_size", 3)
        dilations = c.get("dilations", [1, 2, 4])
        dropout = c.get("dropout", 0.2)
        lr = c.get("lr", 0.001)
        self.epochs = c.get("epochs", 100)
        self.batch_size = c.get("batch_size", 32)
        self.patience = c.get("patience", 15)
        self.n_horizons = n_horizons

        inp = layers.Input(shape=input_shape, name="sequence")
        x = inp

        for i, (f, d) in enumerate(zip(filters, dilations)):
            x = layers.Conv1D(
                filters=f, kernel_size=kernel_size,
                dilation_rate=d, padding="causal",
                activation="relu", name=f"tcn_conv_{i}")(x)
            x = layers.Dropout(dropout)(x)
            if i < len(filters) - 1:
                x = layers.BatchNormalization()(x)

        x = layers.GlobalAveragePooling1D()(x)
        out = layers.Dense(n_horizons, activation="sigmoid", name="hazard")(x)

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
        return "TCN"
