import datetime
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text as text
import tensorflow_datasets as tfds


RUN_ID = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")


def attention_head_assert(hidden_dim):
    return hidden_dim / 64 == 2 or 4 or 8 or 12


def get_model_url(n_layer, hidden_dim, run_custom_model):
    url = None

    if not run_custom_model:
        assert attention_head_assert(hidden_dim) == True, 'Wrong number of attention heads. Ensure hidden_dim is correct.'

        # Replace MODEL_ID value with one of the pre-trained models declared above
        MODEL_ID = 'bert_en_uncased_L-{}_H-{}_A-{}'.format(n_layer, hidden_dim, int(hidden_dim / 64)) 
        MODEL_VERSION = str(2) # the version is from TFHub (last part of the model's URL)

        url = 'https://tfhub.dev/tensorflow/small_bert/{}/{}'.format(MODEL_ID, MODEL_VERSION)
    
    else:

        MODEL_ID = 'custom_bert'


    return url, MODEL_ID


def get_angles(pos, i, d_model):
  angle_rates = 1 / np.power(10000, (2 * (i//2)) / np.float32(d_model))
  return pos * angle_rates


def positional_encoding(position, d_model):
  angle_rads = get_angles(np.arange(position)[:, np.newaxis],
                          np.arange(d_model)[np.newaxis, :],
                          d_model)

  # apply sin to even indices in the array; 2i
  angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])

  # apply cos to odd indices in the array; 2i+1
  angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

  pos_encoding = angle_rads[np.newaxis, ...]

  return tf.cast(pos_encoding, dtype=tf.float32)


def scaled_dot_product_attention(q, k, v, mask):
  """Calculate the attention weights.
  q, k, v must have matching leading dimensions.
  k, v must have matching penultimate dimension, i.e.: seq_len_k = seq_len_v.
  The mask has different shapes depending on its type(padding or look ahead)
  but it must be broadcastable for addition.

  Args:
    q: query shape == (..., seq_len_q, depth)
    k: key shape == (..., seq_len_k, depth)
    v: value shape == (..., seq_len_v, depth_v)
    mask: Float tensor with shape broadcastable
          to (..., seq_len_q, seq_len_k). Defaults to None.

  Returns:
    output, attention_weights
  """

  matmul_qk = tf.matmul(q, k, transpose_b=True)  # (..., seq_len_q, seq_len_k)

  # scale matmul_qk
  dk = tf.cast(tf.shape(k)[-1], tf.float32)
  scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

  # add the mask to the scaled tensor.
  if mask is not None:
    scaled_attention_logits += (mask * -1e9)

  # softmax is normalized on the last axis (seq_len_k) so that the scores
  # add up to 1.
  attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)  # (..., seq_len_q, seq_len_k)

  output = tf.matmul(attention_weights, v)  # (..., seq_len_q, depth_v)

  return output, attention_weights
  

class DataGenerator(tf.keras.utils.Sequence):
    '''
        takes:
            * hypothesis and premise
            * labels
            * batch_size
            * seq_length
            * shuffle
            * include_targets
        
        returns:
            * tuple: ([input_id, input_mask, input_type_id], label) when include_targets=True
            * or list: [input_id, input_mask, input_type_id] when include_targets=False
    '''

    def __init__(
        self,
        hypothesis,
        premise,
        labels,
        batch_size,
        seq_length,
        shuffle=True,
        include_targets=True
    ):
        self.hypothesis = hypothesis
        self.premise = premise
        self.labels = labels
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.shuffle = shuffle
        self.include_targets = include_targets

        self.tokenizer = tf.keras.layers.experimental.preprocessing.TextVectorization(
            max_tokens=self.seq_length,
            output_mode='int',
            output_sequence_length=self.seq_length,
            pad_to_max_tokens=True
        )

        self.tokenizer.adapt(np.char.add(self.hypothesis, self.premise))


        self.indexes = np.arange(len(self.hypothesis))
        self.on_epoch_end()

    def __len__(self):
        # denotes the number of batches per epoch
        return len(self.hypothesis) // self.batch_size

    def __getitem__(self, idx):
        # retrieves the batch of index
        indexes = self.indexes[idx * self.batch_size : (idx + 1) * self.batch_size]
        hypothesis = self.hypothesis[indexes]
        premise = self.premise[indexes]

        # modify tokenized_inputs
        text_inputs = np.char.add(hypothesis, premise)
        tokenized_inputs = self.tokenizer(text_inputs).numpy().astype('float32')

        input_masks = tf.math.equal(tokenized_inputs, 0).numpy().astype('float32')
        input_masks = input_masks[:, tf.newaxis, tf.newaxis, :]

        # return labels depending on whether training/validating
        if self.include_targets:
            labels = np.array(self.labels[indexes], dtype=np.int32)
            return [tokenized_inputs, input_masks, tokenized_inputs], labels
        else:
            return [tokenized_inputs, input_masks, tokenized_inputs]
    
    def on_epoch_end(self):
        # shuffle indexes after each epoch
        if self.shuffle:
            np.random.RandomState(42).shuffle(self.indexes)


class BertSemanticDataGenerator(tf.keras.utils.Sequence):
    '''
        takes:
            * hypothesis and premise
            * labels
            * batch_size
            * seq_length
            * shuffle
            * include_targets
        
        returns:
            * tuple: ([input_id, input_mask, input_type_id], label) when include_targets=True
            * or list: [input_id, input_mask, input_type_id] when include_targets=False
    '''

    def __init__(
        self,
        hypothesis,
        premise,
        labels,
        batch_size,
        seq_length,
        shuffle=True,
        include_targets=True
    ):
        self.hypothesis = hypothesis
        self.premise = premise
        self.labels = labels
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.shuffle = shuffle
        self.include_targets = include_targets

        PREPROCESSOR_URL = 'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3'
        self.preprocessor = hub.load(PREPROCESSOR_URL)

        # tokenizer
        self.tokenizer = hub.KerasLayer(self.preprocessor.tokenize)

        # pack input sequences for the Transformer encoder
        self.bert_pack_inputs = hub.KerasLayer(
            self.preprocessor.bert_pack_inputs,
            arguments=dict(seq_length=self.seq_length)
        )

        self.indexes = np.arange(len(self.hypothesis))
        self.on_epoch_end()

    def __len__(self):
        # denotes the number of batches per epoch
        return len(self.hypothesis) // self.batch_size

    def __getitem__(self, idx):
        # retrieves the batch of index
        indexes = self.indexes[idx * self.batch_size : (idx + 1) * self.batch_size]
        hypothesis = self.hypothesis[indexes]
        premise = self.premise[indexes]

        # modify tokenized_inputs
        text_inputs = [hypothesis, premise]
        tokenized_inputs = [self.tokenizer(segment) for segment in text_inputs]

        # encode
        encoded = self.bert_pack_inputs(tokenized_inputs)

        # convert batch of encoded features to numpy array
        input_ids = np.array(encoded['input_word_ids'], dtype=np.float32)
        input_masks = np.array(encoded['input_mask'], dtype=np.float32)
        input_type_ids = np.array(encoded['input_type_ids'], dtype=np.float32)  # segment_ids ??

        # return labels depending on whether training/validating
        if self.include_targets:
            labels = np.array(self.labels[indexes], dtype=np.int32)
            return [input_ids, input_masks, input_type_ids], labels
        else:
            return [input_ids, input_masks, input_type_ids]
    
    def on_epoch_end(self):
        # shuffle indexes after each epoch
        if self.shuffle:
            np.random.RandomState(42).shuffle(self.indexes)


def get_model_dir(hcc_id, model_id):
    return '/work/cse479/{}/models/snli/{}/{}'.format(hcc_id, model_id, RUN_ID)


def get_cache_dir(hcc_id):
    return '/work/cse479/{}/hubs/snli'.format(hcc_id)


def get_data():
    ds, info = tfds.load('snli', with_info=True, data_dir='./snli_tfds')
    train_df = tfds.as_dataframe(ds['train'])
    valid_df = tfds.as_dataframe(ds['validation'])
    test_df = tfds.as_dataframe(ds['test'])

    train_df = (
        train_df[train_df.label != -1].sample(frac=1.0, random_state=42).reset_index(drop=True)
    )

    valid_df = (
        valid_df[valid_df.label != -1].sample(frac=1.0, random_state=42).reset_index(drop=True)
    )

    test_df = (
        test_df[test_df.label != -1].sample(frac=1.0, random_state=42).reset_index(drop=True)
    )


    return train_df, valid_df, test_df