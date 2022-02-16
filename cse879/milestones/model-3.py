import os
import logging
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text as text
import matplotlib.pyplot as plt

from util import scaled_dot_product_attention, positional_encoding


tf.get_logger().setLevel(logging.ERROR)


class MultiHeadAttention(tf.keras.layers.Layer):
  def __init__(self, d_model, num_heads):
    super(MultiHeadAttention, self).__init__()
    self.num_heads = num_heads
    self.d_model = d_model

    assert d_model % self.num_heads == 0

    self.depth = d_model // self.num_heads

    self.wq = tf.keras.layers.Dense(d_model)
    self.wk = tf.keras.layers.Dense(d_model)
    self.wv = tf.keras.layers.Dense(d_model)

    self.dense = tf.keras.layers.Dense(d_model)

  def split_heads(self, x, batch_size):
    """Split the last dimension into (num_heads, depth).
    Transpose the result such that the shape is (batch_size, num_heads, seq_len, depth)
    """
    x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
    return tf.transpose(x, perm=[0, 2, 1, 3])

  def call(self, v, k, q, mask):
    batch_size = tf.shape(q)[0]

    q = self.wq(q)  # (batch_size, seq_len, d_model)
    k = self.wk(k)  # (batch_size, seq_len, d_model)
    v = self.wv(v)  # (batch_size, seq_len, d_model)

    q = self.split_heads(q, batch_size)  # (batch_size, num_heads, seq_len_q, depth)
    k = self.split_heads(k, batch_size)  # (batch_size, num_heads, seq_len_k, depth)
    v = self.split_heads(v, batch_size)  # (batch_size, num_heads, seq_len_v, depth)

    # scaled_attention.shape == (batch_size, num_heads, seq_len_q, depth)
    # attention_weights.shape == (batch_size, num_heads, seq_len_q, seq_len_k)
    scaled_attention, attention_weights = scaled_dot_product_attention(
        q, k, v, mask)

    scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])  # (batch_size, seq_len_q, num_heads, depth)

    concat_attention = tf.reshape(scaled_attention,
                                  (batch_size, -1, self.d_model))  # (batch_size, seq_len_q, d_model)

    output = self.dense(concat_attention)  # (batch_size, seq_len_q, d_model)

    return output, attention_weights


def point_wise_feed_forward_network(d_model, dff):
  return tf.keras.Sequential([
      tf.keras.layers.Dense(dff, activation='relu'),  # (batch_size, seq_len, dff)
      tf.keras.layers.Dense(d_model)  # (batch_size, seq_len, d_model)
  ])


class EncoderLayer(tf.keras.layers.Layer):
  
  def __init__(self, d_model, num_heads, dff, rate=0.1):
    super(EncoderLayer, self).__init__()

    self.mha = MultiHeadAttention(d_model, num_heads)
    self.ffn = point_wise_feed_forward_network(d_model, dff)

    self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
    self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

    self.dropout1 = tf.keras.layers.Dropout(rate)
    self.dropout2 = tf.keras.layers.Dropout(rate)

  def call(self, x, training, mask):

    attn_output, _ = self.mha(x, x, x, mask)  # (batch_size, input_seq_len, d_model)
    attn_output = self.dropout1(attn_output, training=training)
    out1 = self.layernorm1(x + attn_output)  # (batch_size, input_seq_len, d_model)

    ffn_output = self.ffn(out1)  # (batch_size, input_seq_len, d_model)
    ffn_output = self.dropout2(ffn_output, training=training)
    out2 = self.layernorm2(out1 + ffn_output)  # (batch_size, input_seq_len, d_model)

    return out2


class Encoder(tf.keras.layers.Layer):
  def __init__(self, num_layers, d_model, num_heads, dff, input_vocab_size,
               maximum_position_encoding, rate=0.1):
    super(Encoder, self).__init__()

    self.d_model = d_model
    self.num_layers = num_layers

    self.embedding = tf.keras.layers.Embedding(input_vocab_size, d_model)
    self.pos_encoding = positional_encoding(maximum_position_encoding, self.d_model)

    self.enc_layers = [EncoderLayer(d_model, num_heads, dff, rate)
                       for _ in range(num_layers)]

    self.dropout = tf.keras.layers.Dropout(rate)

  def call(self, x, training, mask):

    seq_len = tf.shape(x)[1]

    # adding embedding and position encoding.
    x = self.embedding(x)  # (batch_size, input_seq_len, d_model)
    x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
    x += self.pos_encoding[:, :seq_len, :]

    x = self.dropout(x, training=training)

    for i in range(self.num_layers):
      x = self.enc_layers[i](x, training, mask)

    return x  # (batch_size, input_seq_len, d_model)


def get_pretrained_m_seq_output(model_url, model_id, input_word_ids, input_mask, input_type_ids):
    bert_model = hub.KerasLayer(model_url, name=model_id)

    # freeze bert_model to reuse pretrained features without modification
    bert_model.trainable = False

    bert_input_dict = {'input_word_ids': input_word_ids, 'input_mask': input_mask, 'input_type_ids': input_type_ids}
    return bert_model(bert_input_dict)['sequence_output'], bert_model


def get_custom_m_seq_output(
    max_length, 
    batch_size, 
    input_word_ids, 
    input_mask,
    num_layers=8
):
    encoder = Encoder(
        num_layers= num_layers,
        d_model=max_length,
        num_heads=int(max_length / 64),
        dff=512,
        input_vocab_size=max_length * batch_size,
        maximum_position_encoding=1000
    )
    
    return encoder(
        x=input_word_ids,
        mask=input_mask
    ), None

def baseline_model(
    batch_size,
    max_length,
    model_id,
    model_url=None,  # returns custom model if model_url is None
    num_layers_custom_bert=8
):
    input_dtype = np.int32 if model_url is not None else np.float32
    # encoded token ids from BERT tokenizer
    input_word_ids = tf.keras.Input(shape=(max_length,), dtype=input_dtype, name='input_word_ids')
    # attention masks (which tokens should be attended to)
    input_mask = tf.keras.Input(shape=(max_length,), dtype=input_dtype, name='input_mask')
    # token type ids == segment ids are binary masks identifying different sequences in the model
    input_type_ids = tf.keras.Input(shape=(max_length,), dtype=input_dtype, name='input_type_ids')

    bert_seq_output, bert_model =\
        get_pretrained_m_seq_output(model_url, model_id, input_word_ids, input_mask, input_type_ids)\
            if model_url is not None else\
                get_custom_m_seq_output(max_length, batch_size, input_word_ids, input_mask, num_layers_custom_bert)

    # other layers
    bi_lstm = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, return_sequences=True))(bert_seq_output)
    avg_pool = tf.keras.layers.GlobalAveragePooling1D()(bi_lstm)
    max_pool = tf.keras.layers.GlobalMaxPooling1D()(bi_lstm)
    concat_hybrid_pool = tf.keras.layers.concatenate([avg_pool, max_pool]) # hybrid pooling approach to bi_lstm output
    dropout = tf.keras.layers.Dropout(0.3)(concat_hybrid_pool)
    output = tf.keras.layers.Dense(3, activation='softmax')(dropout)

    # define functional model
    model = tf.keras.models.Model(
        inputs=[input_word_ids, input_mask, input_type_ids], outputs=output
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    if model_url is not None:
        return model, bert_model
    else:
        return model
