import tensorflow_datasets as tfds

DATA_DIR = './datasets/imdb'

def load_data():
  return tfds.load('imdb_reviews', as_supervised=True, with_info=True, data_dir=DATA_DIR)


def load_splitted_data():
  '''
    returns (validation data, training data), info
  '''

  try:
    return tfds.load(
      'imdb_reviews',
      as_supervised=True,
      with_info=True,
      data_dir=DATA_DIR,
      split=['train[:10%]', 'train[10%:]']
    )
  except Exception as e:
    print('Exception loading tfds:', e)
    ds =  tfds.load(
        'imdb_reviews',
        split=[
            tfds.Split.TRAIN.subsplit(tfds.percent[:10]),
            tfds.Split.TRAIN.subsplit(tfds.percent[10:])
        ],
        with_info=True,
        as_supervised=True
    )
    print('Successfully loaded tfds!')

    return ds
    