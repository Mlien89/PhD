import tensorflow as tf
from util import constant_alpha, ORIGINAL_SHAPE
from util import  EqualizeLearningRate, PixelNormalization, MinibatchSTDDEV

output_activation = tf.keras.activations.tanh
kernel_initializer = 'he_normal'


def upsample_block(x, in_filters, filters, kernel_size=3, strides=1, padding='valid', activation=tf.nn.leaky_relu, name=''):
    '''
        Upsampling + 2 Convolution-Activation
    '''
    upsample = tf.keras.layers.UpSampling2D(size=2, interpolation='nearest')(x)
    upsample_x = EqualizeLearningRate(tf.keras.layers.Conv2D(filters, kernel_size, strides, padding=padding,
                   kernel_initializer=kernel_initializer, bias_initializer='zeros'), name=name+'_conv2d_1')(upsample)
    x = PixelNormalization()(upsample_x)
    x = tf.keras.layers.Activation(activation)(x)
    x = EqualizeLearningRate(tf.keras.layers.Conv2D(filters, kernel_size, strides, padding=padding,
                                   kernel_initializer=kernel_initializer, bias_initializer='zeros'), name=name+'_conv2d_2')(x)
    x = PixelNormalization()(x)
    x = tf.keras.layers.Activation(activation)(x)
    return x, upsample


def downsample_block(x, filters1, filters2, kernel_size=3, strides=1, padding='valid', activation=tf.nn.leaky_relu, name=''):
    '''
        2 Convolution-Activation + Downsampling
    '''
    x = EqualizeLearningRate(tf.keras.layers.Conv2D(filters1, kernel_size, strides, padding=padding,
               kernel_initializer=kernel_initializer, bias_initializer='zeros'), name=name+'_conv2d_1')(x)
    x = tf.keras.layers.Activation(activation)(x)
    x = EqualizeLearningRate(tf.keras.layers.Conv2D(filters2, kernel_size, strides, padding=padding,
               kernel_initializer=kernel_initializer, bias_initializer='zeros'), name=name+'_conv2d_2')(x)
    x = tf.keras.layers.Activation(activation)(x)
    downsample = tf.keras.layers.AveragePooling2D(pool_size=2)(x)

    return downsample


# <h1>Generator</h1>

# In[23]:


def gen_input_block(x):
  
    x = tf.keras.layers.Flatten()(x)
    x = EqualizeLearningRate(
        tf.keras.layers.Dense(16*16*512, kernel_initializer=kernel_initializer, bias_initializer='zeros'),
        name='g_input_dense'
    )(x)
    x = PixelNormalization()(x)
    x = tf.keras.layers.LeakyReLU()(x)
    x = tf.keras.layers.Reshape((16, 16, 512))(x)
    x = EqualizeLearningRate(
        tf.keras.layers.Conv2D(512, 3, strides=1, padding='same', kernel_initializer=kernel_initializer, bias_initializer='zeros'),
        name='g_input_conv2d'
    )(x)
    x = PixelNormalization()(x)
    x = tf.keras.layers.LeakyReLU()(x)
    return x


def _16x16_gen():
    # Initial block
    inputs = tf.keras.Input(shape=(), dtype=tf.float32)
    x = gen_input_block(inputs)
    
    to_rgb = EqualizeLearningRate(
        tf.keras.layers.Conv2D(3, kernel_size=1, strides=1,  padding='same', activation=output_activation,
         kernel_initializer=kernel_initializer, bias_initializer='zeros'),
        name='to_rgb_{}x{}'.format(16, 16)
    )
    
    rgb_out = to_rgb(x)
    model = tf.keras.models.Model(inputs=inputs, outputs=rgb_out)
    return model


def _32x32_gen():
    prev_size = (16, 16)
    curr_size = (32, 32)
    # Initial block
    inputs = tf.keras.Input(shape=(), dtype=tf.float32)
    x = gen_input_block(inputs)

    alpha = tf.keras.layers.Lambda(constant_alpha)(x)

    # Fade in block
    x, up_x = upsample_block(
        x, in_filters=512, filters=512, kernel_size=3, strides=1, padding='same', 
        activation=tf.nn.leaky_relu, name='Up_{}x{}'.format(*curr_size)
    )
    
    
    previous_to_rgb = EqualizeLearningRate(
        tf.keras.layers.Conv2D(3, kernel_size=1, strides=1,  padding='same', activation=output_activation, 
        kernel_initializer=kernel_initializer, bias_initializer='zeros'),
        name='to_rgb_{}x{}'.format(*prev_size)
    )
    to_rgb = EqualizeLearningRate(
        tf.keras.layers.Conv2D(3, kernel_size=1, strides=1,  padding='same', activation=output_activation, 
        kernel_initializer=kernel_initializer, bias_initializer='zeros'), 
        name='to_rgb_{}x{}'.format(*curr_size)
    )

    l_x = to_rgb(x)
    r_x = previous_to_rgb(up_x)
    
    l_x = tf.keras.layers.Multiply()([1 - alpha, l_x])

    r_x = tf.keras.layers.Multiply()([alpha, r_x])
    combined = tf.keras.layers.Add()([l_x, r_x])
    
    outputs = tf.keras.Sequential([
      tf.keras.layers.experimental.preprocessing.Resizing(*ORIGINAL_SHAPE[:2])
    ])(combined)
    
    model = tf.keras.models.Model(inputs=inputs, outputs=outputs)
    return model


def _64x64_gen():
    prev_size = (32, 32)
    curr_size = (64, 64)
    
    # Initial block
    inputs = tf.keras.Input(shape=(), dtype=tf.float32)
    x = gen_input_block(inputs)
    alpha = tf.keras.layers.Lambda(constant_alpha)(x)
    
    # Stable blocks
    x, _ = upsample_block(
        x, in_filters=512, filters=512, kernel_size=3, strides=1, padding='same', activation=tf.nn.leaky_relu, name='Up_{}x{}'.format(*prev_size)
    )
    # Fade in block
    x, up_x = upsample_block(
        x, in_filters=512, filters=512, kernel_size=3, strides=1, padding='same', activation=tf.nn.leaky_relu, name='Up_{}x{}'.format(*curr_size)
    )
    
    previous_to_rgb = EqualizeLearningRate(
        tf.keras.layers.Conv2D(3, kernel_size=1, strides=1,  padding='same', activation=output_activation, 
        kernel_initializer=kernel_initializer, bias_initializer='zeros'), 
        name='to_rgb_{}x{}'.format(*prev_size)
    )
    to_rgb = EqualizeLearningRate(
        tf.keras.layers.Conv2D(3, kernel_size=1, strides=1,  padding='same', activation=output_activation,
            kernel_initializer=kernel_initializer, bias_initializer='zeros'), 
            name='to_rgb_{}x{}'.format(*curr_size)
    )

    l_x = to_rgb(x)
    r_x = previous_to_rgb(up_x)
    
    l_x = tf.keras.layers.Multiply()([1 - alpha, l_x])
    
    r_x = tf.keras.layers.Multiply()([alpha, r_x])
    combined = tf.keras.layers.Add()([l_x, r_x])
    
    outputs = tf.keras.Sequential([
      tf.keras.layers.experimental.preprocessing.Resizing(*ORIGINAL_SHAPE[:2])
    ])(combined)
    
    model = tf.keras.models.Model(inputs=inputs, outputs=outputs)
    return model


def _128x128_gen():
    prev_size_2 = (32, 32)
    prev_size_1 = (64, 64)
    curr_size = (128, 128)
    
    # Initial block
    inputs = tf.keras.Input(shape=(), dtype=tf.float32)
    x = gen_input_block(inputs)
    alpha = tf.keras.layers.Lambda(constant_alpha)(x)
    
    # Stable blocks
    x, _ = upsample_block(
        x, in_filters=512, filters=512, kernel_size=3, strides=1,padding='same', activation=tf.nn.leaky_relu, 
        name='Up_{}x{}'.format(*prev_size_2)
    )
    x, _ = upsample_block(
        x, in_filters=512, filters=512, kernel_size=3, strides=1, padding='same', activation=tf.nn.leaky_relu, 
        name='Up_{}x{}'.format(*prev_size_1)
    )
    
    # Fade in block
    x, up_x = upsample_block(
        x, in_filters=512, filters=512, kernel_size=3, strides=1, padding='same', activation=tf.nn.leaky_relu, 
        name='Up_{}x{}'.format(*curr_size)
    )
    
    previous_to_rgb = EqualizeLearningRate(
        tf.keras.layers.Conv2D(3, kernel_size=1, strides=1,  padding='same', activation=output_activation, kernel_initializer=kernel_initializer, 
            bias_initializer='zeros'), 
        name='to_rgb_{}x{}'.format(*prev_size_1)
    )
    to_rgb = EqualizeLearningRate(
        tf.keras.layers.Conv2D(3, kernel_size=1, strides=1,  padding='same', activation=output_activation, kernel_initializer=kernel_initializer, 
            bias_initializer='zeros'), 
        name='to_rgb_{}x{}'.format(*curr_size)
    )

    l_x = to_rgb(x)
    r_x = previous_to_rgb(up_x)

    l_x = tf.keras.layers.Multiply()([1 - alpha, l_x])
    
    r_x = tf.keras.layers.Multiply()([alpha, r_x])
    combined = tf.keras.layers.Add()([l_x, r_x])
    
    outputs = tf.keras.Sequential([
      tf.keras.layers.experimental.preprocessing.Resizing(*ORIGINAL_SHAPE[:2])
    ])(combined)
    
    model = tf.keras.models.Model(inputs=inputs, outputs=outputs)
    return model


def _256x256_gen():
    prev_size_3 = (32, 32)
    prev_size_2 = (64, 64)
    prev_size_1 = (128, 128)
    curr_size = (256, 256)
    
    # Initial block
    inputs = tf.keras.Input(shape=(), dtype=tf.float32)
    x = gen_input_block(inputs)
    alpha = tf.keras.layers.Lambda(constant_alpha)(x)
    
    # Stable blocks
    x, _ = upsample_block(
        x, in_filters=512, filters=512, kernel_size=3, strides=1, padding='same', activation=tf.nn.leaky_relu, 
        name='Up_{}x{}'.format(*prev_size_3)
    )
    x, _ = upsample_block(
        x, in_filters=512, filters=512, kernel_size=3, strides=1, padding='same', activation=tf.nn.leaky_relu, 
        name='Up_{}x{}'.format(*prev_size_2)
    )
    x, _ = upsample_block(
        x, in_filters=512, filters=512, kernel_size=3, strides=1, padding='same', activation=tf.nn.leaky_relu, 
        name='Up_{}x{}'.format(*prev_size_1)
    )
    
    # Fade in block
    x, up_x = upsample_block(
        x, in_filters=512, filters=256, kernel_size=3, strides=1, padding='same', activation=tf.nn.leaky_relu, 
        name='Up_{}x{}'.format(*curr_size)
    )
    
    previous_to_rgb = EqualizeLearningRate(
        tf.keras.layers.Conv2D(3, kernel_size=1, strides=1,  padding='same', activation=output_activation,
            kernel_initializer=kernel_initializer, bias_initializer='zeros'), 
        name='to_rgb_{}x{}'.format(*prev_size_1)
    )
    to_rgb = EqualizeLearningRate(
        tf.keras.layers.Conv2D(3, kernel_size=1, strides=1,  padding='same', activation=output_activation, 
        kernel_initializer=kernel_initializer, bias_initializer='zeros'), 
        name='to_rgb_{}x{}'.format(*curr_size)
    )
    
    l_x = to_rgb(x)
    r_x = previous_to_rgb(up_x)
     
    l_x = tf.keras.layers.Multiply()([1 - alpha, l_x])
     
    r_x = tf.keras.layers.Multiply()([alpha, r_x])
    combined = tf.keras.layers.Add()([l_x, r_x])
    
    outputs = tf.keras.Sequential([
      tf.keras.layers.experimental.preprocessing.Resizing(*ORIGINAL_SHAPE[:2])
    ])(combined)
    
    model = tf.keras.models.Model(inputs=inputs, outputs=outputs)
    return model



# <h1>Discriminator</h1>

# In[62]:


def dis_output_block(x):
    x = MinibatchSTDDEV()(x)
    x = EqualizeLearningRate(tf.keras.layers.Conv2D(512, 3, strides=1, padding='same',
                                    kernel_initializer=kernel_initializer, bias_initializer='zeros'), name='d_output_conv2d_1')(x)
    x = tf.keras.layers.LeakyReLU()(x)
    x = EqualizeLearningRate(tf.keras.layers.Conv2D(512, 4, strides=1, padding='valid',
                                    kernel_initializer=kernel_initializer, bias_initializer='zeros'), name='d_output_conv2d_2')(x)
    x = tf.keras.layers.LeakyReLU()(x)
    x = tf.keras.layers.Flatten()(x)
    x = EqualizeLearningRate(tf.keras.layers.Dense(1, kernel_initializer=kernel_initializer, bias_initializer='zeros', activation='sigmoid'), name='d_output_dense')(x)
    
    return x


def _16x16_dis(): 
    inputs = tf.keras.Input(shape=((16, 16, 3)), dtype=tf.float32)
    
    # From RGB
    from_rgb = EqualizeLearningRate(
        tf.keras.layers.Conv2D(512, kernel_size=1, strides=1, padding='same', activation=tf.nn.leaky_relu, kernel_initializer=kernel_initializer, 
            bias_initializer='zeros'), 
        name='from_rgb_{}x{}'.format(16, 16)
    )
    x = from_rgb(inputs)
    x = EqualizeLearningRate(
        tf.keras.layers.Conv2D(512, kernel_size=1, strides=1, padding='same', activation=tf.nn.leaky_relu, 
            kernel_initializer=kernel_initializer, bias_initializer='zeros'), 
        name='conv2d_up_channel'
    )(x)
    x = dis_output_block(x)
    model = tf.keras.models.Model(inputs=inputs, outputs=x)
    return model


def _32x32_dis():
    
    prev_size = (16, 16)
    curr_size = (32, 32)
    
    fade_in_channel = 512
    origin_inputs = tf.keras.Input(ORIGINAL_SHAPE)
    inputs = tf.keras.Sequential([
      tf.keras.layers.experimental.preprocessing.Resizing(*curr_size)
    ])(origin_inputs)
    alpha = tf.keras.layers.Lambda(constant_alpha)(inputs)
    downsample = tf.keras.layers.AveragePooling2D(pool_size=2)
     
    previous_from_rgb = EqualizeLearningRate(
        tf.keras.layers.Conv2D(512, kernel_size=1, strides=1, padding='same', activation=tf.nn.leaky_relu, kernel_initializer=kernel_initializer, 
            bias_initializer='zeros'), 
        name='from_rgb_{}x{}'.format(*prev_size)
    )
    l_x = previous_from_rgb(downsample(inputs))
    l_x = tf.keras.layers.Multiply()([1 - alpha, l_x])
     
    from_rgb = EqualizeLearningRate(
        tf.keras.layers.Conv2D(512, kernel_size=1, strides=1, padding='same', activation=tf.nn.leaky_relu, kernel_initializer=kernel_initializer, 
            bias_initializer='zeros'), 
        name='from_rgb_{}x{}'.format(*curr_size)
    )
    r_x = from_rgb(inputs)
    
    # Fade in block
    r_x = downsample_block(r_x, filters1=512, filters2=fade_in_channel, kernel_size=3, strides=1,
                                            padding='same', activation=tf.nn.leaky_relu, name='Down_{}x{}'.format(*curr_size))
    r_x = tf.keras.layers.Multiply()([alpha, r_x])
    x = tf.keras.layers.Add()([l_x, r_x])

    # Stable block
    x = dis_output_block(x)
    model = tf.keras.models.Model(inputs=origin_inputs, outputs=x)
    return model


def _64x64_dis():
    
    prev_size = (32, 32)
    curr_size = (64, 64)
    
    fade_in_channel = 512
    origin_inputs = tf.keras.Input(ORIGINAL_SHAPE)
    inputs = tf.keras.Sequential([
      tf.keras.layers.experimental.preprocessing.Resizing(*curr_size)
    ])(origin_inputs)
    alpha = tf.keras.layers.Lambda(constant_alpha)(inputs)
    downsample = tf.keras.layers.AveragePooling2D(pool_size=2)
    
    
    previous_from_rgb = EqualizeLearningRate(
        tf.keras.layers.Conv2D(512, kernel_size=1, strides=1, padding='same', activation=tf.nn.leaky_relu, kernel_initializer=kernel_initializer, 
                bias_initializer='zeros'), 
            name='from_rgb_{}x{}'.format(*prev_size)
        )
    l_x = previous_from_rgb(downsample(inputs))
    l_x = tf.keras.layers.Multiply()([1 - alpha, l_x])
     
    from_rgb = EqualizeLearningRate(
        tf.keras.layers.Conv2D(512, kernel_size=1, strides=1, padding='same', activation=tf.nn.leaky_relu, kernel_initializer=kernel_initializer, 
            bias_initializer='zeros'), 
        name='from_rgb_{}x{}'.format(*curr_size)
    )
    r_x = from_rgb(inputs)
    
    # Fade in block
    r_x = downsample_block(
        r_x, filters1=512, filters2=fade_in_channel, kernel_size=3, strides=1, padding='same', activation=tf.nn.leaky_relu, 
        name='Down_{}x{}'.format(*curr_size)
    )
    r_x = tf.keras.layers.Multiply()([alpha, r_x])
    x = tf.keras.layers.Add()([l_x, r_x])
    
    # Stable blocks
    x = downsample_block(
        x, filters1=512, filters2=512, kernel_size=3, strides=1, padding='same', activation=tf.nn.leaky_relu, 
        name='Down_{}x{}'.format(*prev_size)
    )
    x = dis_output_block(x)
    model = tf.keras.models.Model(inputs=origin_inputs, outputs=x)
    return model


def _128x128_dis():
    prev_size_2 = (32, 32)
    prev_size_1 = (64, 64)
    curr_size = (128, 128)
    
    fade_in_channel = 512
    origin_inputs = tf.keras.Input(ORIGINAL_SHAPE)
    inputs = tf.keras.Sequential([
      tf.keras.layers.experimental.preprocessing.Resizing(*curr_size)
    ])(origin_inputs)
    alpha = tf.keras.layers.Lambda(constant_alpha)(inputs)
    downsample = tf.keras.layers.AveragePooling2D(pool_size=2)
     
    previous_from_rgb = EqualizeLearningRate(
        tf.keras.layers.Conv2D(512, kernel_size=1, strides=1, padding='same', activation=tf.nn.leaky_relu, kernel_initializer=kernel_initializer, 
            bias_initializer='zeros'), 
        name='from_rgb_{}x{}'.format(*prev_size_1)
    )
    l_x = previous_from_rgb(downsample(inputs))
    l_x = tf.keras.layers.Multiply()([1 - alpha, l_x])
     
    from_rgb = EqualizeLearningRate(
        tf.keras.layers.Conv2D(512, kernel_size=1, strides=1, padding='same', activation=tf.nn.leaky_relu, kernel_initializer=kernel_initializer, 
            bias_initializer='zeros'), 
        name='from_rgb_{}x{}'.format(*curr_size)
    )
    r_x = from_rgb(inputs)
    
    # Fade in block
    r_x = downsample_block(
        r_x, filters1=512, filters2=fade_in_channel, kernel_size=3, strides=1, padding='same', activation=tf.nn.leaky_relu, 
        name='Down_{}x{}'.format(*curr_size)
    )
    r_x = tf.keras.layers.Multiply()([alpha, r_x])
    x = tf.keras.layers.Add()([l_x, r_x])
    
    # Stable blocks
    x = downsample_block(
        x, filters1=512, filters2=512, kernel_size=3, strides=1, padding='same', activation=tf.nn.leaky_relu, 
        name='Down_{}x{}'.format(*prev_size_1)
    )
    x = downsample_block(
        x, filters1=512, filters2=512, kernel_size=3, strides=1, padding='same', activation=tf.nn.leaky_relu, 
        name='Down_{}x{}'.format(*prev_size_2)
    )
    x = dis_output_block(x)
    model = tf.keras.models.Model(inputs=origin_inputs, outputs=x)
    return model


def _256x256_dis():
    
    prev_size_3 = (32, 32)
    prev_size_2 = (64, 64)
    prev_size_1 = (128, 128)
    curr_size = (256, 256)
    
    fade_in_channel = 512
    origin_inputs = tf.keras.Input(ORIGINAL_SHAPE)
    inputs = tf.keras.Sequential([
      tf.keras.layers.experimental.preprocessing.Resizing(*curr_size)
    ])(origin_inputs)
    alpha = tf.keras.layers.Lambda(constant_alpha)(inputs)
    downsample = tf.keras.layers.AveragePooling2D(pool_size=2)

    previous_from_rgb = EqualizeLearningRate(
        tf.keras.layers.Conv2D(512, kernel_size=1, strides=1, padding='same', activation=tf.nn.leaky_relu, kernel_initializer=kernel_initializer, 
            bias_initializer='zeros'), 
        name='from_rgb_{}x{}'.format(*prev_size_1)
    )
    l_x = previous_from_rgb(downsample(inputs))
    l_x = tf.keras.layers.Multiply()([1 - alpha, l_x])
   
   
    from_rgb = EqualizeLearningRate(
        tf.keras.layers.Conv2D(256, kernel_size=1, strides=1, padding='same', activation=tf.nn.leaky_relu, kernel_initializer=kernel_initializer, 
            bias_initializer='zeros'), 
        name='from_rgb_{}x{}'.format(*curr_size)
    )
    r_x = from_rgb(inputs)
    
    # Fade in block
    r_x = downsample_block(
        r_x, filters1=256, filters2=fade_in_channel, kernel_size=3, strides=1, padding='same', activation=tf.nn.leaky_relu, 
        name='Down_{}x{}'.format(*curr_size)
    )
    r_x = tf.keras.layers.Multiply()([alpha, r_x])
    x = tf.keras.layers.Add()([l_x, r_x])
    
    # Stable blocks
    x = downsample_block(
        x, filters1=512, filters2=512, kernel_size=3, strides=1, padding='same', activation=tf.nn.leaky_relu, 
        name='Down_{}x{}'.format(*prev_size_1)
    )
    x = downsample_block(
        x, filters1=512, filters2=512, kernel_size=3, strides=1, padding='same', activation=tf.nn.leaky_relu, 
        name='Down_{}x{}'.format(*prev_size_2)
    )

    x = downsample_block(
        x, filters1=512, filters2=512, kernel_size=3, strides=1, padding='same', activation=tf.nn.leaky_relu, 
        name='Down_{}x{}'.format(*prev_size_3)
    )
    x = dis_output_block(x)
    model = tf.keras.models.Model(inputs=origin_inputs, outputs=x)
    return model


# In[ ]: