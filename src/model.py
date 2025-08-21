from __future__ import annotations
import tensorflow as tf
from tensorflow.keras import layers, Model, Input

def build_caption_model(vocab_size: int, max_len: int, feature_dim: int = 2048, embed_dim: int = 256, lstm_units: int = 256) -> Model:
    feat_in = Input(shape=(feature_dim,), name="image_features")
    h = layers.Dense(lstm_units, activation="relu", name="h_from_img")(feat_in)
    c = layers.Dense(lstm_units, activation="relu", name="c_from_img")(feat_in)

    seq_in = Input(shape=(max_len,), dtype="int32", name="seq_in")
    x = layers.Embedding(vocab_size, embed_dim, mask_zero=True, name="emb")(seq_in)
    x = layers.LSTM(lstm_units, return_sequences=True, name="lstm")(x, initial_state=[h, c])
    x = layers.TimeDistributed(layers.Dense(vocab_size, activation="softmax"), name="logits")(x)

    model = Model(inputs=[feat_in, seq_in], outputs=x, name="cnn_lstm_captioner")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=2e-4),
        loss="sparse_categorical_crossentropy",
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name="acc")]
    )
    return model

def build_inception_encoder():
    base = tf.keras.applications.InceptionV3(include_top=False, weights="imagenet", pooling="avg")
    base.trainable = False
    preprocess = tf.keras.applications.inception_v3.preprocess_input
    return base, preprocess