import numpy as np
import tensorflow as tf
from tensorflow.keras import layers


class TransformerBlock(layers.Layer):
    """Single transformer encoder block."""

    def __init__(self, d_model, n_heads, ff_dim, dropout=0.1, **kwargs):
        super().__init__(**kwargs)
        self.attn = layers.MultiHeadAttention(
            num_heads=n_heads, key_dim=d_model // n_heads)
        self.ffn = tf.keras.Sequential([
            layers.Dense(ff_dim, activation="relu"),
            layers.Dense(d_model),
        ])
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(dropout)
        self.dropout2 = layers.Dropout(dropout)

    def call(self, inputs, training=False):
        attn_out = self.attn(inputs, inputs)
        attn_out = self.dropout1(attn_out, training=training)
        out1 = self.layernorm1(inputs + attn_out)
        ffn_out = self.ffn(out1)
        ffn_out = self.dropout2(ffn_out, training=training)
        return self.layernorm2(out1 + ffn_out)


class TransformerHazard:
    """Transformer-based hazard model for multihorizon failure prediction."""

    def __init__(self, input_shape, n_horizons, config=None):
        c = config or {}
        d_model = c.get("d_model", 32)
        n_heads = c.get("n_heads", 4)
        n_layers = c.get("n_layers", 2)
        ff_dim = c.get("ff_dim", 64)
        dropout = c.get("dropout", 0.1)
        lr = c.get("lr", 0.001)
        self.epochs = c.get("epochs", 100)
        self.batch_size = c.get("batch_size", 32)
        self.patience = c.get("patience", 15)
        self.n_horizons = n_horizons

        # Project input to d_model if needed
        inp = layers.Input(shape=input_shape, name="sequence")
        x = layers.Dense(d_model, activation="relu")(inp)
        x = layers.PositionalEncoding()(x)  # custom layer defined below

        for i in range(n_layers):
            x = TransformerBlock(
                d_model, n_heads, ff_dim, dropout,
                name=f"transformer_{i}")(x)

        x = layers.GlobalAveragePooling1D()(x)
        x = layers.Dropout(dropout)(x)
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
        return "Transformer"


# Custom positional encoding layer
@tf.keras.utils.register_keras_serializable()
class PositionalEncoding(layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, inputs):
        seq_len = tf.shape(inputs)[1]
        d_model = tf.shape(inputs)[2]
        pos = tf.range(seq_len, dtype=tf.float32)[:, tf.newaxis]
        i = tf.range(d_model, dtype=tf.float32)[tf.newaxis, :]
        angle_rates = 1 / tf.pow(
            10000.0, (2 * (i // 2)) / tf.cast(d_model, tf.float32))
        angle_rads = pos * angle_rates
        sines = tf.sin(angle_rads[:, 0::2])
        cosines = tf.cos(angle_rads[:, 1::2])
        pe = tf.concat([sines, cosines], axis=-1)[tf.newaxis, :, :]
        return inputs + pe
