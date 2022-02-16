
import tensorflow as tf
import tensorflow_datasets as tfds
from functools import partial


DATA_DIR = './datasets/imdb'


class AttentionLayer(tf.keras.layers.Layer):

    def __init__(self, filters=100, **kwargs):

        super(AttentionLayer, self).__init__(**kwargs)

        self.query_layer = tf.keras.layers.Conv1D(
            filters=filters,
            kernel_size=4,
            padding='same',
            name='QueryLayer'
        )

        self.value_layer = tf.keras.layers.Conv1D(
            filters=filters,
            kernel_size=4,
            padding='same',
            name='ValueLayer'
        )

        self.attention_layer = tf.keras.layers.Attention()
    
    def call(self, inputs):

        query = self.query_layer(inputs)
        value = self.value_layer(inputs)

        attention = self.attention_layer([query, value])

        return tf.keras.layers.concatenate([query, attention])


def sequential_model(
  dense_units=[250],
  n_dense=1,
  learning_rate=1e-3,
  lstm_dropout_rate=0.2,
  n_lstm=1,
  lstm_units=[128],
  max_tokens=1024,
  vocab_size=2048,
  embedding_size=128,
  mask_zero=True,
  n_conv1d=1
):

  ds = tfds.load('imdb_reviews', as_supervised=True, data_dir=DATA_DIR)
  train_dataset = ds['train'].batch(64).prefetch(1)

  text_vectorizer = tf.keras.layers.experimental.preprocessing.TextVectorization(
    max_tokens=max_tokens, name='TextVectorizationLayer'
  )
  text_vectorizer.adapt(train_dataset.map(lambda text, label: text))

  model = tf.keras.models.Sequential()

  model.add(tf.keras.Input(shape=(), dtype=tf.string, name='InputLayer'))
  model.add(text_vectorizer)
  model.add(tf.keras.layers.Embedding(vocab_size, embedding_size, mask_zero=mask_zero, name='EmbeddingLayer'))
  model.add(AttentionLayer(name='AttentionLayer'))

  for i, layer in enumerate(range(n_conv1d)):
    model.add(tf.keras.layers.Conv1D(filters=32, kernel_size=4, padding='same', activation='relu', name='Conv1DLayer{}'.format(i + 1)))
  
  model.add(tf.keras.layers.MaxPooling1D(pool_size=2, name='MaxPoolLayer1'))

  for i, layer in enumerate(range(n_lstm)):
    model.add(tf.keras.layers.LSTM(lstm_units[i], dropout=lstm_dropout_rate, name='LSTMLayer{}'.format(i + 1)))

  for i, layer in enumerate(range(n_dense)):  
    model.add(tf.keras.layers.Dense(dense_units[i], activation='relu', name='DenseLayer{}'.format(i + 1)))

  model.add(tf.keras.layers.Dense(1, activation='sigmoid', name='OutputLayer'))

  optimizer = tf.keras.optimizers.Adam(lr=learning_rate)
  model.compile(
      loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
      optimizer=optimizer,
      metrics=["accuracy"]
  )


  return model


def params_to_search():

  dense_units=[
    (250,),
    (256, 128),
    (256, 128, 64),
    (512, 256, 128, 64)
  ]
  n_dense=[1, 2, 3, 4]
  learning_rate=[1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
  lstm_dropout_rate=[0.1, 0.2, 0.3, 0.4, 0.5]
  n_lstm=[1, 2, 3]
  lstm_units=[
    (128,),
    (128, 64),
    (512, 128, 64)
  ]
  max_tokens=[1000, 1024, 2048]
  vocab_size=[2000, 2048, 4096]
  embedding_size=[20, 70, 128, 256]
  mask_zero=[True, False]
  n_conv1d=[1, 2, 3]

  return dict(
        dense_units=dense_units,
        n_dense=n_dense,
        learning_rate=learning_rate,
        lstm_dropout_rate=lstm_dropout_rate,
        n_lstm=n_lstm,
        lstm_units=lstm_units,
        max_tokens=max_tokens,
        vocab_size=vocab_size,
        embedding_size=embedding_size,
        mask_zero=mask_zero,
        n_conv1d=n_conv1d
  )



def model_wrapped_by_sklearn():
  return tf.keras.wrappers.scikit_learn.KerasClassifier(sequential_model)



DefaultConv1D = partial(tf.keras.layers.Conv1D, kernel_size=3, strides=1, padding="SAME", use_bias=False)

class ResidualUnit(tf.keras.layers.Layer):
    def __init__(self, filters, activation = "selu", **kwargs):
        super(ResidualUnit, self).__init__(**kwargs)
        self.filters = filters
        self.strides = 1
        self.activation = tf.keras.activations.get(activation)
        self.main_layers = [
            DefaultConv1D(self.filters, strides=self.strides),
            tf.keras.layers.Dropout(0.2),
            self.activation,
            DefaultConv1D(self.filters),
            tf.keras.layers.Dropout(0.2)]
        self.skip_layers = []
        if self.strides > 1:
            self.skip_layers = [
                DefaultConv1D(self.filters, kernel_size=1, strides=self.strides),
                tf.keras.layers.Dropout(0.2)]

    def call(self, inputs, filters, strides):
        self.filters = filters
        self.strides = strides
        
        Z = inputs
        for layer in self.main_layers:
            Z = layer(Z)
        skip_Z = inputs
        for layer in self.skip_layers:
            skip_Z = layer(skip_Z)
        return self.activation(Z + skip_Z)


class ResidualModel(tf.keras.models.Model):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.inputs = tf.keras.layers.InputLayer(input_shape=())

        self.shallower_layers = [
            tf.keras.layers.LSTM(8, dropout=0.4, return_sequences=True),
            tf.keras.layers.LSTM(8, recurrent_dropout=0.2, return_sequences=True),
        ]
        
        self.previous_filters = 8
        self.residual_unit = ResidualUnit(self.previous_filters)
        self.filter_sequence = [8] * 3 + [4] * 4 + [2] * 6 + [8] * 3
        self.strides = 1

        self.later_layers = [
            tf.keras.layers.Conv1D(filters=8, kernel_size=4, padding='same'),
            tf.keras.layers.LeakyReLU(alpha=0.2),
            tf.keras.layers.MaxPool1D(pool_size=3, strides=2, padding='same'),
            tf.keras.layers.GlobalAvgPool1D(),
            tf.keras.layers.Dense(16),
            tf.keras.layers.Dropout(rate=0.3),
            tf.keras.layers.Dense(8)
        ]


    def call(self, X):

        X = self.inputs(X)

        for layer in self.shallower_layers:
            X = layer(X)
        
        # add residual units: 3, 4, 6, 3
        for filters in self.filter_sequence:
            self.strides = 1 if filters == self.previous_filters else 2
            X = self.residual_unit(inputs=X, filters=filters, strides=self.strides)
            self.previous_filters = filters
        
        for layer in self.later_layers:
            X = layer(X)

        return X


def model_with_stacked_attention_layers():

  ds = tfds.load('imdb_reviews', as_supervised=True, data_dir=DATA_DIR)
  train_dataset = ds['train'].batch(64).prefetch(1)

  text_vectorizer = tf.keras.layers.experimental.preprocessing.TextVectorization(
    max_tokens=1024, name='TextVectorizationLayer'
  )
  text_vectorizer.adapt(train_dataset.map(lambda text, label: text))

  model = tf.keras.models.Sequential([
    tf.keras.Input(shape=(), name='InputLayer', dtype=tf.string),
    text_vectorizer,
    tf.keras.layers.Embedding(
        input_dim=2048,
        output_dim=32,
        mask_zero=True,
        name='EmbeddingLayer'
    ),
    AttentionLayer(filters=64, name='AttentionLayer1'),
    AttentionLayer(filters=64, name='AttentionLayer2'),
    AttentionLayer(filters=64, name='AttentionLayer3'),
    ResidualModel(),
    tf.keras.layers.Dense(1, activation='sigmoid', name='OutputLayer')
  ])

  optimizer = tf.keras.optimizers.Adam()
  model.compile(
      loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
      optimizer=optimizer,
      metrics=["accuracy"]
  )


  return model

