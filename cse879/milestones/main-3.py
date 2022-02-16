import os
import logging
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text as text
import matplotlib.pyplot as plt
from model import baseline_model
from util import BertSemanticDataGenerator, DataGenerator
from util import get_model_url, get_model_dir, get_cache_dir, get_data


tf.get_logger().setLevel(logging.ERROR)


run_custom_model = True


# From a set of 24 pre-trained BERT models on TFHub,
# we incorporated those represented with "o" in our model

#       | H128  | H256  | H512  | H768   
# --------------------------------------
# L2    |   o   |   o   |   o   |   o
# --------------------------------------
# L4    |   o   |   o   |       |
# --------------------------------------
# L6    |   o   |       |       |
# --------------------------------------
# L8    |   o   |       |       |
# --------------------------------------
# L10   |   o   |       |       |
# --------------------------------------
# L12   |   o   |       |       |

# Replace these values to pick the model to run
hidden_dim = 128
layer_num = 2

MODEL_URL, MODEL_ID = get_model_url(layer_num, hidden_dim, run_custom_model)

HCC_USERNAME = 'kshamavu2'  # replace this with your HCC username
MODEL_DIR = get_model_dir(HCC_USERNAME, MODEL_ID)
TFHUB_CACHE_DIR = get_cache_dir(HCC_USERNAME)
os.environ['TFHUB_CACHE_DIR'] = TFHUB_CACHE_DIR

train_df, valid_df, test_df = get_data()

y_train = tf.keras.utils.to_categorical(train_df.label, num_classes=3)
y_val = tf.keras.utils.to_categorical(valid_df.label, num_classes=3)
y_test = tf.keras.utils.to_categorical(test_df.label, num_classes=3)

batch_size = 32

def run_model_with_pretrained_bert(epochs=2):
    max_length = 128
    
    model, bert_layer = baseline_model(batch_size, max_length, MODEL_ID, MODEL_URL)

    model.summary()

    train_data = BertSemanticDataGenerator(
        train_df["hypothesis"].values.astype("str"),
        train_df["premise"].values.astype("str"),
        y_train,
        seq_length=max_length,
        batch_size=batch_size,
        shuffle=True,
    )
    valid_data = BertSemanticDataGenerator(
        valid_df["hypothesis"].values.astype("str"),
        valid_df["premise"].values.astype("str"),
        y_val,
        seq_length=max_length,
        batch_size=batch_size,
        shuffle=False,
    )


    model.fit(
        train_data,
        validation_data=valid_data,
        epochs=epochs,
        use_multiprocessing=True,
        workers=-1,
    )


    print("Unfreezing the bert_layer for fine-tuning...")
    bert_layer.trainable = True

    # Recompile the model to make the change effective.
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-5),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    model.summary()

    checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(MODEL_DIR + '/saved_model', save_best_only=True)
    tensorboard_cb = tf.keras.callbacks.TensorBoard(log_dir=MODEL_DIR)

    model.fit(
        train_data,
        validation_data=valid_data,
        epochs=epochs,
        use_multiprocessing=True,
        workers=-1,
        callbacks=[checkpoint_cb, tensorboard_cb]
    )


def run_model_with_custom_bert(epochs=30):
    max_length = 256

    model = baseline_model(batch_size, max_length, MODEL_ID, model_url=None, num_layers_custom_bert=8)

    model.summary()

    train_data = DataGenerator(
        train_df["hypothesis"].values.astype("str"),
        train_df["premise"].values.astype("str"),
        y_train,
        seq_length=max_length,
        batch_size=batch_size,
        shuffle=True,
    )
    valid_data = DataGenerator(
        valid_df["hypothesis"].values.astype("str"),
        valid_df["premise"].values.astype("str"),
        y_val,
        seq_length=max_length,
        batch_size=batch_size,
        shuffle=False,
    )

    checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(MODEL_DIR + '/saved_model', save_best_only=True)
    tensorboard_cb = tf.keras.callbacks.TensorBoard(log_dir=MODEL_DIR)

    model.fit(
        train_data,
        validation_data=valid_data,
        epochs=epochs,
        use_multiprocessing=True,
        workers=-1,
        callbacks=[checkpoint_cb, tensorboard_cb]
    )



if run_custom_model:
    run_model_with_custom_bert()
else:
    run_model_with_pretrained_bert()
    