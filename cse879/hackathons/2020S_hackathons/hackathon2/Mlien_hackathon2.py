#!/usr/bin/env python
# coding: utf-8

# # Hackathon #2
# 
# Mason Lien

# ### Homework
# 
# Your homework is to specify a network with `tf.keras.layers`, train it on the MNIST dataset (as above, but with train/validation split), and try out 2 or 3 variations of different architectures. I.e., change the number of neurons or layers, change the activation function (you can find more in the documentation at [`tf.nn`](https://www.tensorflow.org/api_docs/python/tf/nn)), or even change the optimizer ([`tf.keras.optimizers`](https://www.tensorflow.org/api_docs/python/tf/keras/optimizers)). Write up a paragraph or two with your observations. E.g., how did it affect the final accuracy on the validation data? How did it affect the rate at which the model improved? Remember to add early stopping and increase the number of training epochs. Submit a `.pdf` with the writeup and `.py` with the code.
# 
# I'm expecting this to take about an hour (or less if you're experienced). Feel free to use any code from this or previous hackathons. If you don't understand how to do any part of this or if it's taking you longer than that, please let me know in office hours or by email (both can be found on the syllabus). I'm also happy to discuss if you just want to ask more questions about anything in this notebook!

# In[1]:


# We'll start with our library imports...
from __future__ import print_function

import numpy as np                 # to use numpy arrays
import tensorflow as tf            # to specify and run computation graphs
import tensorflow_datasets as tfds # to load training data
import matplotlib.pyplot as plt    # to visualize data and draw plots
from tqdm import tqdm              # to track progress of loops
import time


# In[47]:


# The first 90% of the training data
# Use this data for the training loop
train = tfds.load('mnist', split='train[:90%]').shuffle(1024).batch(32)

# And the last 10%, we'll hold out as the validation set
# Notice the python-style indexing, but in a string and with percentages
# After the training loop, run another loop over this data without the gradient updates to calculate accuracy
validation = tfds.load('mnist', split='train[-10%:]').shuffle(1024).batch(32)


# In[55]:


# Model 1
model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(100, activation = tf.nn.relu))
model.add(tf.keras.layers.Dense(10))
optimizer = tf.keras.optimizers.Nadam()

train_loss_values = []
train_accuracy_values = []
validate_loss_values = []
validate_accuracy_values = []
# Loop through data for each epoch
epochs = 5
for epoch in range(epochs):
    for batch in tqdm(train):
        with tf.GradientTape() as tape:
            # run network
            x = tf.reshape(tf.cast(batch['image'], tf.float32), [-1, 784])
            labels = batch['label']
            logits = model(x)

            # calculate loss
            train_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels)    
            train_loss_values.append(train_loss)
    
        # gradient update
        grads = tape.gradient(train_loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
    
        # calculate accuracy
        predictions = tf.argmax(logits, axis=1)
        accuracy = tf.reduce_mean(tf.cast(tf.equal(predictions, labels), tf.float32))
        train_accuracy_values.append(accuracy)
    print("Train Accuracy:", np.mean(train_accuracy_values))
    
    time.sleep(1.0)
    
    for batch in tqdm(validation):
        val_x = tf.reshape(tf.cast(batch['image'], tf.float32), [-1, 784])
        val_labels = batch['label']
        val_logits = model(val_x, training = False)
        
        validate_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=val_logits, labels=val_labels)
        validate_loss_values.append(validate_loss)

        val_predictions = tf.argmax(val_logits, axis=1)
        val_accuracy = tf.reduce_mean(tf.cast(tf.equal(val_predictions, val_labels), tf.float32))
        validate_accuracy_values.append(val_accuracy)
    print("Validate Accuracy:", np.mean(validate_accuracy_values))   

    time.sleep(1.0)
    
print(model.summary())    
# accuracy
print("Train Accuracy:", np.mean(train_accuracy_values))
print("Validate Accuracy", np.mean(validate_accuracy_values))


# In[36]:


#plot per-datum loss
#train_loss_values = np.concatenate(train_loss_values)
#validate_loss_values = np.concatenate(validate_loss_values)
plt.figure()
plt.plot(train_loss_values, 'b', label = 'Training Loss')
plt.plot(validate_loss_values, 'r', label = 'Validate Loss')
plt.title('Training and Validation Loss')
plt.legend()


# In[50]:


# Model 2
model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(128, activation = tf.nn.relu))
model.add(tf.keras.layers.Dense(128, activation = tf.nn.relu))
model.add(tf.keras.layers.Dense(10))
optimizer = tf.keras.optimizers.Adam()

train_loss_values = []
train_accuracy_values = []
validate_loss_values = []
validate_accuracy_values = []
# Loop through data for each epoch
epochs = 5
for epoch in range(epochs):
    for batch in tqdm(train):
        with tf.GradientTape() as tape:
            # run network
            x = tf.reshape(tf.cast(batch['image'], tf.float32), [-1, 784])
            labels = batch['label']
            logits = model(x)

            # calculate loss
            train_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels)    
            train_loss_values.append(train_loss)
    
        # gradient update
        grads = tape.gradient(train_loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
    
        # calculate accuracy
        predictions = tf.argmax(logits, axis=1)
        accuracy = tf.reduce_mean(tf.cast(tf.equal(predictions, labels), tf.float32))
        train_accuracy_values.append(accuracy)
    print("Train Accuracy:", np.mean(train_accuracy_values))
    
    time.sleep(1.0)
    
    for batch in tqdm(validation):
        val_x = tf.reshape(tf.cast(batch['image'], tf.float32), [-1, 784])
        val_labels = batch['label']
        val_logits = model(val_x, training = False)
        
        validate_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=val_logits, labels=val_labels)
        validate_loss_values.append(validate_loss)

        val_predictions = tf.argmax(val_logits, axis=1)
        val_accuracy = tf.reduce_mean(tf.cast(tf.equal(val_predictions, val_labels), tf.float32))
        validate_accuracy_values.append(val_accuracy)
    print("Validate Accuracy:", np.mean(validate_accuracy_values))   

    time.sleep(1.0)
    
print(model.summary())    
# accuracy
print("Train Accuracy:", np.mean(train_accuracy_values))
print("Validate Accuracy", np.mean(validate_accuracy_values))


# In[56]:


# Model 3
model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(128, activation = tf.nn.sigmoid))
model.add(tf.keras.layers.Dense(128, activation = tf.nn.sigmoid))
model.add(tf.keras.layers.Dense(10))
optimizer = tf.keras.optimizers.Adam()

train_loss_values = []
train_accuracy_values = []
validate_loss_values = []
validate_accuracy_values = []
# Loop through data for each epoch
epochs = 5
for epoch in range(epochs):
    for batch in tqdm(train):
        with tf.GradientTape() as tape:
            # run network
            x = tf.reshape(tf.cast(batch['image'], tf.float32), [-1, 784])
            labels = batch['label']
            logits = model(x)

            # calculate loss
            train_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels)    
            train_loss_values.append(train_loss)
    
        # gradient update
        grads = tape.gradient(train_loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
    
        # calculate accuracy
        predictions = tf.argmax(logits, axis=1)
        accuracy = tf.reduce_mean(tf.cast(tf.equal(predictions, labels), tf.float32))
        train_accuracy_values.append(accuracy)
    print("Train Accuracy:", np.mean(train_accuracy_values))
    
    time.sleep(1.0)
    
    for batch in tqdm(validation):
        val_x = tf.reshape(tf.cast(batch['image'], tf.float32), [-1, 784])
        val_labels = batch['label']
        val_logits = model(val_x, training = False)
        
        validate_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=val_logits, labels=val_labels)
        validate_loss_values.append(validate_loss)

        val_predictions = tf.argmax(val_logits, axis=1)
        val_accuracy = tf.reduce_mean(tf.cast(tf.equal(val_predictions, val_labels), tf.float32))
        validate_accuracy_values.append(val_accuracy)
    print("Validate Accuracy:", np.mean(validate_accuracy_values))   

    time.sleep(1.0)
    
print(model.summary())    
# accuracy
print("Train Accuracy:", np.mean(train_accuracy_values))
print("Validate Accuracy", np.mean(validate_accuracy_values))


# In[ ]:




