import numpy as np
import tensorflow as tf
from tensorflow.keras import layers


class ContinuousHazard:
    """Hazard model supporting continuous (arbitrary) service horizons.

    Instead of training separate heads for each discrete H, the model
    accepts horizon H as an input, enabling prediction at any H value
    at inference time.

    Architecture:  [features] ──► encoder ──► concat ──► MLP ──► sigmoid
                    [horizon]  ──► embedding ──►
    """

    def __init__(self, feature_dim, config=None):
        c = config or {}
        d_model = c.get("d_model", 32)
        ff_dim = c.get("ff_dim", 64)
        dropout = c.get("dropout", 0.2)
        lr = c.get("lr", 0.001)
        self.epochs = c.get("epochs", 100)
        self.batch_size = c.get("batch_size", 32)
        self.patience = c.get("patience", 15)

        # Feature encoder
        feat_inp = layers.Input(shape=(feature_dim,), name="features")
        x = layers.Dense(d_model, activation="relu")(feat_inp)
        x = layers.Dropout(dropout)(x)
        x = layers.Dense(d_model, activation="relu")(x)
        feat_encoded = layers.Dropout(dropout)(x)

        # Horizon embedding
        h_inp = layers.Input(shape=(1,), name="horizon")
        h_emb = layers.Dense(d_model // 2, activation="relu")(h_inp)

        # Concatenate
        merged = layers.Concatenate()([feat_encoded, h_emb])
        merged = layers.Dense(ff_dim, activation="relu")(merged)
        merged = layers.Dropout(dropout)(merged)
        merged = layers.Dense(ff_dim // 2, activation="relu")(merged)
        out = layers.Dense(1, activation="sigmoid", name="hazard")(merged)

        self.model = tf.keras.Model(inputs=[feat_inp, h_inp], outputs=out)
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(lr),
            loss="binary_crossentropy",
            metrics=["auc"])

    def fit(self, X_feat, X_h, y, X_val_feat=None, X_val_h=None, y_val=None):
        callbacks = [tf.keras.callbacks.EarlyStopping(
            monitor="val_loss" if X_val_feat is not None else "loss",
            patience=self.patience, restore_best_weights=True,
            verbose=0)]
        val_data = None
        if X_val_feat is not None:
            val_data = ([X_val_feat, X_val_h], y_val)
        self.model.fit(
            [X_feat, X_h], y,
            validation_data=val_data,
            epochs=self.epochs, batch_size=self.batch_size,
            callbacks=callbacks, verbose=0)
        return self

    def predict_proba(self, X_feat, X_h):
        return self.model.predict([X_feat, X_h], verbose=0).ravel()

    @property
    def name(self):
        return "ContinuousHazard"
