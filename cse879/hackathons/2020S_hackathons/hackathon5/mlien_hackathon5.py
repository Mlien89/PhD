#!/usr/bin/env python
# coding: utf-8

# Mason Lien

# ### Homework
# 
# Please do two things for this one:
# 1. Write a few sentences about the differences between the `ResBlock` and `Bottleneck` layers above. Why might the Bottleneck block be more suitable for deeper architectures with more layers?
# 2. Write some python code which builds a network using the general structure described above and either ResBlock or Bottleneck blocks. It doesn't have to be a full set of code that runs, just a function or class that builds a network from these blocks. You might find this architecture useful for homework 1.
# 
# On Canvas, submit your python code in a `.py` and your short write-up in a `.txt` or `.pdf`.
# 
# I'm expecting this to take about an hour (or less if you're experienced). Feel free to use any code from this or previous hackathons. If you don't understand how to do any part of this or if it's taking you longer than that, please let me know in office hours or by email (both can be found on the syllabus). I'm also happy to discuss if you just want to ask more questions about anything in this notebook!

# In[2]:


# We'll start with our library imports...
from __future__ import print_function

import numpy as np                 # to use numpy arrays
import tensorflow as tf            # to specify and run computation graphs
import tensorflow_datasets as tfds # to load training data
import matplotlib.pyplot as plt    # to visualize data and draw plots
from tqdm import tqdm              # to track progress of loops


# In[191]:


class ResBlock(tf.keras.layers.Layer):
    def __init__(self, num_channels, num_residuals, first_block=False,
                 **kwargs):
        super(ResBlock, self).__init__(**kwargs)
        self.residual_layers = []
        for i in range(num_residuals):
            if i == 0 and not first_block:
                self.residual_layers.append(
                    Residual(num_channels, use_1x1conv=True, strides=2))
            else:
                self.residual_layers.append(Residual(num_channels))

    def call(self, X):
        for layer in self.residual_layers.layers:
            X = layer(X)
        return X


# In[199]:


model = tf.keras.Sequential()
model.add(tf.keras.layers.Conv2D(64, kernel_size=7, strides=2, padding='same'))
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.Activation('relu'))
model.add(tf.keras.layers.MaxPool2D(pool_size=3, strides=2, padding='same'))
model.add(ResBlock(64, 2, first_block=True))
model.add(ResBlock(128, 2))
model.add(ResBlock(256, 2))
model.add(tf.keras.layers.GlobalAvgPool2D())
model.add(tf.keras.layers.Dense(units=10))


# In[200]:


x = tf.random.uniform((1, 256, 256, 1))
y = model(x)
model.summary()

