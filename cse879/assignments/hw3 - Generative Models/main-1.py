#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os


# In[2]:


import time


# In[3]:


import logging


# In[4]:


import numpy as np


# In[5]:


import tensorflow as tf


# In[6]:


from functools import partial


# In[7]:


import matplotlib.pyplot as plt


# In[8]:


import tensorflow_datasets as tfds


# In[9]:


tf.get_logger().setLevel(logging.ERROR)
logging.getLogger('matplotlib').setLevel(level=logging.CRITICAL)


# In[10]:


from model import _16x16_gen, _32x32_gen, _64x64_gen, _128x128_gen, _256x256_gen


# In[11]:


from model import _16x16_dis, _32x32_dis, _64x64_dis, _128x128_dis, _256x256_dis


# In[12]:


from util import preprocess_image, print_status_bar, calculate_batch_size, plot_multiple_images


# In[ ]:





# In[13]:


CLEAN_RUN = str(3)


# In[14]:


HCC_USR = 'username'  # replace this value


# In[15]:


PRE_PATH = '/work/cse479/'


# In[16]:


DATA_DIR = PRE_PATH + HCC_USR + '/datasets/celeb_a'


# In[17]:


MODEL_DIR = PRE_PATH + HCC_USR + '/models/celeb-' + CLEAN_RUN + '/'


# In[18]:


LOG_DIR = PRE_PATH + HCC_USR + '/models/celeb-' + CLEAN_RUN + '-logs/'


# In[19]:


AUTO_TUNE = tf.data.experimental.AUTOTUNE


# In[20]:


N_MODELS = 5
N_EPOCHS = 25  # i.e. each model will train for N_EPOCHS / N_MODELS epochs


# In[ ]:





# In[21]:


resume_training = False


# In[22]:


image_size = 16  # start with 16x16


# In[23]:


batch_size = 32


# In[ ]:





# In[24]:


# create dirs
try:
    os.makedirs(MODEL_DIR)
except Exception as e:
    print(e)
# Creates a file writer for the log directory.
file_writer = tf.summary.create_file_writer(LOG_DIR)


# In[ ]:





# In[25]:

# Only works with tensorflow-datasets==4.2.0
data, info = tfds.load('celeb_a', with_info=True, data_dir=DATA_DIR)


# In[26]:


TRAIN_SIZE = info.splits['train'].num_examples


# In[ ]:





# In[27]:


target_size = None if resume_training and image_size != 16 else image_size
preprocess_function = partial(preprocess_image, target_size=target_size)


# In[28]:


train_data = data['train'].map(preprocess_function).shuffle(1024).batch(batch_size, drop_remainder=True).prefetch(tf.data.experimental.AUTOTUNE)


# In[29]:


sample_img = next(iter(train_data))


# In[30]:


plt.imshow(sample_img[0])


# In[ ]:





# In[31]:


def model_builder(tr):
    generator = None
    discriminator = None
    print('Getting {}x{} model'.format(tr, tr))
    if tr == 16:
        generator = _16x16_gen()
        discriminator = _16x16_dis()
    elif tr == 32:
        generator = _32x32_gen()
        discriminator = _32x32_dis()
    elif tr == 64:
        generator = _64x64_gen()
        discriminator = _64x64_dis()
    elif tr == 128:
        generator = _128x128_gen()
        discriminator = _128x128_dis()
    elif tr == 256:
        generator = _256x256_gen()
        discriminator = _256x256_dis()
    else:
        print('Target resolution (tr) models are not defined...')
        
    return generator, discriminator


# In[32]:


generator, discriminator = model_builder(image_size)


# In[33]:


if resume_training:
    curr_size = (image_size, image_size)
    g_path = os.path.join(MODEL_DIR, '{}x{}_generator.h5'.format(*curr_size))
    d_path = os.path.join(MODEL_DIR, '{}x{}_discriminator.h5'.format(*curr_size))

    if os.path.isfile(g_path) and os.path.isfile(d_path): 
        generator.load_weights(g_path, by_name=False)
        discriminator.load_weights(d_path, by_name=False)
        
        print('resuming training for {}x{} model'.format(*curr_size))


# In[34]:


gan = tf.keras.models.Sequential([generator, discriminator])


# In[35]:


discriminator.compile(loss="binary_crossentropy", optimizer="rmsprop")
discriminator.trainable = False
gan.compile(loss="binary_crossentropy", optimizer="rmsprop")


# In[ ]:





# In[ ]:





# In[36]:


def train_pggan(
    gan, 
    dataset, 
    train_size, 
    image_size,
    n_pggan=5,
    n_epochs=15
):
    switch_res_every_n_epoch = n_epochs / n_pggan  # epochs per pggan
    
    generator, discriminator = gan.layers
    
    for epoch in range(1, n_epochs + 1):
        print('Epoch {}/{}'.format(epoch, n_epochs))
        
        if image_size >= 512:
            print('\nResolution reached 256 * 256 -> Finished training')
            break
            
        l_disc = []
        l_gen = []
        eta = []
        tic = time.time()
        
        print('Current resolution: {}x{}'.format(image_size, image_size))
        
        step = 0
        for X_batch in dataset:
            
            batch_size = X_batch.shape[0]
            
            step += 1
            
            # phase 1 - training the discriminator
            
            noise = tf.random.normal(shape=[batch_size])
            
            generated_images = generator(noise, training=True)
            
            X_fake_and_real = tf.concat([generated_images, tf.cast(X_batch, dtype=tf.float32)], axis=0)
            
            y1 = tf.constant([[0.]] * batch_size + [[1.]] * batch_size)
            
            discriminator.trainable = True
            
            disc_ = discriminator.train_on_batch(X_fake_and_real, y1, return_dict=True)
            
            # phase 2 - training the generator
            
            noise = tf.random.normal(shape=[batch_size])
            
            y2 = tf.constant([[1.]] * batch_size)
            
            # discriminator.trainable = False  
            # "All existing layers in both networks remain trainable throughout the training process"
            
            gen_ = gan.train_on_batch(noise, y2, return_dict=True)

            l_disc.append(disc_['loss'])
            l_gen.append(gen_['loss'])
            eta.append(time.time() - tic)
            
            print_status_bar(
                step * (batch_size), 
                train_size, 
                [{'eta(s)': np.mean(eta)}], 
                [{'d_loss': np.mean(l_disc)}],
                [{'g_loss': np.mean(l_gen)}]
            )
            
            # write loss to tensorboard
            if step % 10 == 0:
                with file_writer.as_default():
                    tf.summary.scalar('G_loss', np.mean(l_gen), step=step)
                    tf.summary.scalar('D_loss', np.mean(l_disc), step=step)
        
        print_status_bar(
            train_size, 
            train_size, 
            [{'eta(s)': np.mean(eta)}], 
            [{'d_loss': np.mean(l_disc)}],
            [{'g_loss': np.mean(l_gen)}]
        )
        
        # saving weights @ each epoch
        print('Saving {}x{} model'.format(image_size, image_size))
        g, d = gan.layers
        curr_size = (image_size, image_size)
        g_path = os.path.join(MODEL_DIR, '{}x{}_generator.h5'.format(*curr_size))

        d_path = os.path.join(MODEL_DIR, '{}x{}_discriminator.h5'.format(*curr_size))

        g.save_weights(g_path)
        d.save_weights(d_path)
        
        # peek at generated images
        plot_multiple_images(generated_images, 8)
        
        # swithing to another model
        if epoch % switch_res_every_n_epoch == 0:
            previous_image_size = int(image_size)
            image_size = int(image_size * 2)
            
            if image_size > 256:
                print('\nResolution reached 256 * 256 -> Finished training')
                break
            
            generator, discriminator = model_builder(image_size)
            
            # load previous model's weights
            prev_size = (previous_image_size , previous_image_size)
            prev_g_path = os.path.join(MODEL_DIR, '{}x{}_generator.h5'.format(*prev_size))

            prev_d_path = os.path.join(MODEL_DIR, '{}x{}_discriminator.h5'.format(*prev_size))
            
            print('Copying weights from previous {}x{} model'.format(*prev_size))
            generator.load_weights(prev_g_path, by_name=True)
            discriminator.load_weights(prev_d_path, by_name=True)
            
            gan = tf.keras.models.Sequential([generator, discriminator])
            
            discriminator.compile(loss="binary_crossentropy", optimizer="rmsprop")
            discriminator.trainable = False
            gan.compile(loss="binary_crossentropy", optimizer="rmsprop")
            
            # preprocess and re-batch data
            batch_size = calculate_batch_size(image_size)
            preprocess_function = partial(preprocess_image, target_size=None)
            dataset = data['train'].map(preprocess_function).shuffle(1024).batch(batch_size, drop_remainder=True).prefetch(tf.data.experimental.AUTOTUNE)
            
            print('\n\n--------- ********** ---------')
            print('Training {}x{} model'.format(image_size, image_size))


# In[ ]:


train_pggan(
    gan, 
    train_data, 
    TRAIN_SIZE, 
    image_size,
    n_pggan=N_MODELS,
    n_epochs=N_EPOCHS
)


# In[ ]:





# In[ ]:





# In[ ]:




