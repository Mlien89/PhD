import logging
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


K = tf.keras.backend
ORIGINAL_SHAPE = (218, 178, 3)
tf.get_logger().setLevel(logging.ERROR)
logging.getLogger('matplotlib').setLevel(level=logging.CRITICAL)


def normalize(image):
    '''
        normalizing the images to [0, 1]
    '''
    image = tf.cast(image, tf.float32)
    image = (image) / 255.0
    return image

def augmentation(image):
    '''
        Perform some augmentation
    '''
    image = tf.image.random_flip_left_right(image)
    return image

def preprocess_image(images, target_size=128):
    images = images['image']
    
    target_size = ORIGINAL_SHAPE[:2] if target_size is None else (target_size, target_size)
    images = tf.image.resize(images, target_size, method='nearest', antialias=True)
    images = augmentation(images)
    images = normalize(images)

    return images


class EqualizeLearningRate(tf.keras.layers.Wrapper):

    def __init__(self, layer, **kwargs):
        super(EqualizeLearningRate, self).__init__(layer, **kwargs)
        self._track_trackable(layer, name='layer')
        self.is_rnn = isinstance(self.layer, tf.keras.layers.RNN)

    def build(self, input_shape):
        """Build `Layer`"""
        input_shape = tf.TensorShape(input_shape)
        self.input_spec = tf.keras.layers.InputSpec(
            shape=[None] + input_shape[1:])

        if not self.layer.built:
            self.layer.build(input_shape)

        kernel_layer = self.layer.cell if self.is_rnn else self.layer

        if not hasattr(kernel_layer, 'kernel'):
            raise ValueError('`EqualizeLearningRate` must wrap a layer that'
                             ' contains a `kernel` for weights')

        if self.is_rnn:
            kernel = kernel_layer.recurrent_kernel
        else:
            kernel = kernel_layer.kernel

        # He constant
        self.fan_in, self.fan_out= self._compute_fans(kernel.shape)
        self.he_constant = tf.Variable(1.0 / np.sqrt(self.fan_in), dtype=tf.float32, trainable=False)

        self.v = kernel
        self.built = True
    
    def call(self, inputs, training=True):
        """Call `Layer`"""

        with tf.name_scope('compute_weights'):
            # Multiply the kernel with the he constant.
            kernel = tf.identity(self.v * self.he_constant)
            
            if self.is_rnn:
                print(self.is_rnn)
                self.layer.cell.recurrent_kernel = kernel
                update_kernel = tf.identity(self.layer.cell.recurrent_kernel)
            else:
                self.layer.kernel = kernel
                update_kernel = tf.identity(self.layer.kernel)

            # Ensure we calculate result after updating kernel.
            with tf.control_dependencies([update_kernel]):
                outputs = self.layer(inputs)
                return outputs

    def compute_output_shape(self, input_shape):
        return tf.TensorShape(
            self.layer.compute_output_shape(input_shape).as_list())
    
    def _compute_fans(self, shape, data_format='channels_last'):
        
        if len(shape) == 2:
            fan_in = shape[0]
            fan_out = shape[1]
        elif len(shape) in {3, 4, 5}:
            # Assuming convolution kernels (1D, 2D or 3D).
            # TH kernel shape: (depth, input_depth, ...)
            # TF kernel shape: (..., input_depth, depth)
            if data_format == 'channels_first':
                receptive_field_size = np.prod(shape[2:])
                fan_in = shape[1] * receptive_field_size
                fan_out = shape[0] * receptive_field_size
            elif data_format == 'channels_last':
                receptive_field_size = np.prod(shape[:-2])
                fan_in = shape[-2] * receptive_field_size
                fan_out = shape[-1] * receptive_field_size
            else:
                raise ValueError('Invalid data_format: ' + data_format)
        else:
            # No specific assumptions.
            fan_in = np.sqrt(np.prod(shape))
            fan_out = np.sqrt(np.prod(shape))
        return fan_in, fan_out



class PixelNormalization(tf.keras.layers.Layer):
    
    def __init__(self):
        super(PixelNormalization, self).__init__()
        self.epsilon = 1e-8

    def call(self, inputs):
        return inputs / tf.sqrt(tf.reduce_mean(tf.square(inputs), axis=-1, keepdims=True) + self.epsilon)
    
    def compute_output_shape(self, input_shape):
        return input_shape

    
    
class MinibatchSTDDEV(tf.keras.layers.Layer):
   
    def __init__(self):
        super(MinibatchSTDDEV, self).__init__()
        self.group_size = 4

    def call(self, inputs):
        group_size = tf.minimum(self.group_size, tf.shape(inputs)[0])     # Minibatch must be divisible by (or smaller than) group_size.
        s = inputs.shape                                             # [NHWC]  Input shape.
        y = tf.reshape(inputs, [group_size, -1, s[1], s[2], s[3]])   # [GMHWC] Split minibatch into M groups of size G.
        y = tf.cast(y, tf.float32)                              # [GMHWC] Cast to FP32.
        y -= tf.reduce_mean(y, axis=0, keepdims=True)           # [GMHWC] Subtract mean over group.
        y = tf.reduce_mean(tf.square(y), axis=0)                # [MHWC]  Calc variance over group.
        y = tf.sqrt(y + 1e-8)                                   # [MHWC]  Calc stddev over group.
        y = tf.reduce_mean(y, axis=[1,2,3], keepdims=True)      # [M111]  Take average over fmaps and pixels.
        y = tf.cast(y, inputs.dtype)                                 # [M111]  Cast back to original data type.
        y = tf.tile(y, [group_size, s[1], s[2], 1])             # [NHW1]  Replicate over group and pixels.
        return tf.concat([inputs, y], axis=-1)                        # [NHWC]  Append as new fmap.
    
    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1], input_shape[2], input_shape[3] + 1)
    
    
def constant_alpha(input_batch):
    tf_constant = K.constant(np.array([0.65]).reshape((1, 1)).astype(np.float32))
    batch_size = K.shape(input_batch)[0]
    tiled_constant = K.tile(tf_constant, (batch_size, 1))

    return tiled_constant


def plot_multiple_images(images, n_cols=None):
    n_cols = n_cols or len(images)
    n_rows = (len(images) - 1) // n_cols + 1
    if images.shape[-1] == 1:
        images = np.squeeze(images, axis=-1)
    plt.figure(figsize=(n_cols, n_rows))
    for index, image in enumerate(images):
        plt.subplot(n_rows, n_cols, index + 1)
        plt.imshow(image, cmap="binary")
        plt.axis("off")

        
def progress_bar(iteration, total, size=30):
    running = iteration < total
    c = '>' if running else '='
    p = (size) * iteration // total
    fmt = '{{:-{}d}}/{{}} [{{}}]'.format(len(str(total)))
    params = [iteration, total, '=' * p + c + '.' * (size - p - 1)]
    return fmt.format(*params)


def print_status_bar(iteration, total, eta=[], loss=[], metrics=[], size=30):
    metrics = ' - '.join(['{}: {:.4f}'.format(k, v[k]) for v in eta + loss + metrics for k in v.keys()])
    end = '' if iteration < total else '\n'
    print('\r{} - {}'.format(progress_bar(iteration, total), metrics), end=end)


def calculate_batch_size(image_size):
    if image_size == 16:
        return 32
    elif image_size == 32:
        return 16
    elif image_size == 64:
        return 8
    else:
        return 4

   
