#!/usr/bin/env python
# coding: utf-8

# Mason Lien
# 
# ### Homework
# 
# Re-write your code from hackathon 2 to use convolutional layers and add code to plot a confusion matrix on the validation data.
# 
# Specifically, write code to calculate a confusion matrix of the model output on the validation data, and compare to the true labels to calculate a confusion matrix with [tf.math.confusion_matrix](https://www.tensorflow.org/api_docs/python/tf/math/confusion_matrix). (For the inexperienced, [what is a confusion matrix?](https://en.wikipedia.org/wiki/Confusion_matrix)) Use the code example from [scikit-learn](https://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html) to help visualise the confusion matrix if you'd like as well.
# 
# On Canvas, submit your python code in a `.py` and your confusion matrix in a `.png` or `.txt`.
# 
# I'm expecting this to take about an hour (or less if you're experienced). Feel free to use any code from this or previous hackathons. If you don't understand how to do any part of this or if it's taking you longer than that, please let me know in office hours or by email (both can be found on the syllabus). I'm also happy to discuss if you just want to ask more questions about anything in this notebook!

# In[1]:


from __future__ import print_function
import tensorflow as tf
import numpy as np
import pandas as pd
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
from tqdm import tqdm
from tensorflow.keras import layers, models


# In[2]:


train_set, test_set = tfds.load(name='cifar10', split =['train', 'test'])


# In[12]:


x_train = []
y_train = []

for example in tfds.as_numpy(train_set):
    new_img = example['image']
    x_train.append(new_img)
    y_train.append(example['label'])


# In[13]:


x_train = np.asarray(x_train).astype("float32")
y_train = np.asarray(y_train).astype("float32")

print('X_train.shape =',x_train.shape)
print('y_train.shape =',y_train.shape)


# In[14]:


x_val = x_train[-5000:]
y_val = y_train[-5000:]

x_train = x_train[:-5000]
y_train = y_train[:-5000]


# In[6]:


#X_val, X_train = X_val / 255., X_train / 255.


# In[15]:


input_shape = (32, 32, 3)

x_train=x_train.reshape(x_train.shape[0], x_train.shape[1], x_train.shape[2], 3)
x_train=x_train / 255.0
x_val = x_val.reshape(x_val.shape[0], x_val.shape[1], x_val.shape[2], 3)
x_val=x_val / 255.0


# In[16]:


model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10, activation = 'softmax'))
model.summary()


# In[17]:


batch_size = 32
num_classes = 10
epochs = 2


# In[18]:


y_train = tf.one_hot(y_train.astype(np.int32), depth=10)
y_val = tf.one_hot(y_val.astype(np.int32), depth=10)


# In[19]:


model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

history = model.fit(x_train, y_train, batch_size = batch_size, epochs=2, 
                    validation_data=(x_val, y_val))


# In[20]:


val_loss, val_acc = model.evaluate(x_val, y_val)


# In[27]:


# Predict the values from the validation dataset
y_pred = model.predict(x_val)
# Convert predictions classes to one hot vectors 
y_pred_classes = np.argmax(y_pred,axis = 1) 
# Convert validation observations to one hot vectors
y_true = np.argmax(y_val,axis = 1)
# compute the confusion matrix
confusion_mtx = tf.math.confusion_matrix(y_true, y_pred_classes)
print(confusion_mtx)

