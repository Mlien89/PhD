from sklearn.model_selection import cross_val_score, RandomizedSearchCV, GridSearchCV

import tensorflow as tf

import numpy as np

from functools import partial

from model import sequential_model, params_to_search, model_wrapped_by_sklearn, model_with_stacked_attention_layers

from util import load_data, load_splitted_data



model_to_run = 1  # change this value to select which model to run
# 1 == best model
# 2 == grid search
# 3 == model with stacked attention
# 4 == randomized search

"""<h1>Best Model</h1>"""

def run_best_model():
  (valid_set_raw, train_set_raw), info = load_splitted_data()

  train_batch_size = valid_batch_size = 32
  train_set = train_set_raw.shuffle(10000).batch(train_batch_size).prefetch(1)
  valid_set = valid_set_raw.batch(valid_batch_size).prefetch(1)

  model = sequential_model()
  model.summary()
  model.fit(train_set, validation_data=valid_set, epochs=5)





"""<h1>Grid Search</h1>"""

def run_grid_search():

  (valid_set_raw, train_set_raw), info = load_splitted_data()

  train_batch_size = valid_batch_size = 32
  train_set = train_set_raw.shuffle(10000).batch(train_batch_size).prefetch(1)
  valid_set = valid_set_raw.batch(valid_batch_size).prefetch(1)

  train_size, valid_size = len(list(train_set_raw.as_numpy_iterator())), len(list(valid_set_raw.as_numpy_iterator()))
  print('train_size:', train_size, 'valid_size:', valid_size)

  X_train, y_train = np.concatenate([x for x, y in train_set], axis=0), np.concatenate([y for x, y in train_set], axis=0)
  X_valid, y_valid = np.concatenate([x for x, y in valid_set], axis=0), np.concatenate([y for x, y in valid_set], axis=0)

  keras_clf = model_wrapped_by_sklearn()

  grid_search_cv = GridSearchCV(
      keras_clf,
      params_to_search(),
      cv=3
  )

  checkpoint_cb = tf.keras.callbacks.ModelCheckpoint('./imdb_grid_search/saved_model', save_best_only=True)

  grid_search_cv.fit(
      X_train, y_train,
      validation_data=(X_valid, y_valid),
      epochs=30,
      callbacks=[checkpoint_cb]
  )

  best_params_, best_score_ = grid_search_cv.best_params_,  grid_search_cv.best_score_
  print('best_params_:', best_params_, 'best_score_:', best_score_)


"""<h1>Randomized Search</h1>"""

def run_rand_search():

  (valid_set_raw, train_set_raw), info = load_splitted_data()

  train_batch_size = valid_batch_size = 32
  train_set = train_set_raw.shuffle(10000).batch(train_batch_size).prefetch(1)
  valid_set = valid_set_raw.batch(valid_batch_size).prefetch(1)

  train_size, valid_size = len(list(train_set_raw.as_numpy_iterator())), len(list(valid_set_raw.as_numpy_iterator()))
  print('train_size:', train_size, 'valid_size:', valid_size)

  X_train, y_train = np.concatenate([x for x, y in train_set], axis=0), np.concatenate([y for x, y in train_set], axis=0)
  X_valid, y_valid = np.concatenate([x for x, y in valid_set], axis=0), np.concatenate([y for x, y in valid_set], axis=0)

  keras_clf = model_wrapped_by_sklearn()

  rnd_search_cv = RandomizedSearchCV(
        keras_clf,
        params_to_search(),
        cv=3,
        n_iter=8
    )

  checkpoint_cb = tf.keras.callbacks.ModelCheckpoint('./imdb_rnd_search/saved_model', save_best_only=True)

  rnd_search_cv.fit(
      X_train, y_train,
      validation_data=(X_valid, y_valid),
      epochs=30,
      callbacks=[checkpoint_cb]
  )

  best_params_, best_score_ = rnd_search_cv.best_params_,  rnd_search_cv.best_score_
  print('best_params_:', best_params_, 'best_score_:', best_score_)



"""<h1>Model With Stacked Attention</h1>"""

def run_model_with_stacked_attention():
  (valid_set_raw, train_set_raw), info = load_splitted_data()

  train_batch_size = valid_batch_size = 32
  train_set = train_set_raw.shuffle(10000).batch(train_batch_size).prefetch(1)
  valid_set = valid_set_raw.batch(valid_batch_size).prefetch(1)

  model = model_with_stacked_attention_layers()
  model.summary()
  model.fit(train_set, validation_data=valid_set, epochs=5)



if model_to_run == 1:
  run_best_model()
elif model_to_run == 2:
  run_grid_search()
elif model_to_run == 3:
  run_model_with_stacked_attention()
elif model_to_run == 4:
  run_rand_search()
else:
  print('Select model to run...')


