# https://github.com/ShahzaibWaseem/SpatialPyramidPooling_tf2/blob/master/SpatialPyramidPooling.py
import tensorflow as tf
from tensorflow.keras.layers import Layer
import tensorflow.keras.backend as K

class SpatialPyramidPooling(Layer):
  """Spatial pyramid pooling layer for 2D inputs.
  See Spatial Pyramid Pooling in Deep Convolutional Networks for Visual Recognition,
  K. He, X. Zhang, S. Ren, J. Sun
  # Arguments
    pool_list: list of int
      List of pooling regions to use. The length of the list is the number of pooling regions,
      each int in the list is the number of regions in that pool. For example [1,2,4] would be 3
      regions with 1, 2x2 and 4x4 max pools, so 21 outputs per feature map
  # Input shape
    4D tensor with shape:
    `(samples, channels, rows, cols)` if dim_ordering='channels_first'
    or 4D tensor with shape:
    `(samples, rows, cols, channels)` if dim_ordering='channels_last'.
  # Output shape
    2D tensor with shape:
    `(samples, channels * sum([i * i for i in pool_list])`
  """

  def __init__(self, pool_list, **kwargs):
    self.dim_ordering = K.image_data_format()
    assert self.dim_ordering in {'channels_last', 'channels_first'}, \
      'dim_ordering must be in {channels_last, channels_first}'
    
    self.pool_list = pool_list
    self.num_outputs_per_channel = sum([i * i for i in pool_list])

    super(SpatialPyramidPooling, self).__init__(**kwargs)

  def build(self, input_shape):
    if self.dim_ordering == 'channels_first':
      self.nb_channels = input_shape[1]
    elif self.dim_ordering == 'channels_last':
      self.nb_channels = input_shape[3]
    super(SpatialPyramidPooling, self).build(input_shape)

  def compute_output_shape(self, input_shape):
    return (input_shape[0], self.nb_channels * self.num_outputs_per_channel)

  def call(self, x, mask=None):
    input_shape = tf.shape(x)

    if self.dim_ordering == 'channels_first':
      num_rows = input_shape[2]
      num_cols = input_shape[3]
      axis = (2,3)
    elif self.dim_ordering == 'channels_last':
      num_rows = input_shape[1]
      num_cols = input_shape[2]
      axis = (1,2)

    outputs = []
    
    for pool_num, num_pool_regions in enumerate(self.pool_list):
      h_step = tf.cast(num_rows, tf.float32) / tf.cast(num_pool_regions, tf.float32)
      w_step = tf.cast(num_cols, tf.float32) / tf.cast(num_pool_regions, tf.float32)
      
      for jy in range(num_pool_regions):
        for ix in range(num_pool_regions):
          # Use proper floating point division and ensure non-zero sizes
          y1 = tf.cast(tf.math.floor(tf.cast(jy, tf.float32) * h_step), tf.int32)
          y2 = tf.cast(tf.math.ceil(tf.cast(jy + 1, tf.float32) * h_step), tf.int32)
          x1 = tf.cast(tf.math.floor(tf.cast(ix, tf.float32) * w_step), tf.int32)
          x2 = tf.cast(tf.math.ceil(tf.cast(ix + 1, tf.float32) * w_step), tf.int32)
          
          # Ensure minimum size of 1x1
          y2 = tf.maximum(y2, y1 + 1)
          x2 = tf.maximum(x2, x1 + 1)
          
          # Clamp to valid ranges
          y1 = tf.maximum(0, tf.minimum(y1, num_rows - 1))
          y2 = tf.maximum(y1 + 1, tf.minimum(y2, num_rows))
          x1 = tf.maximum(0, tf.minimum(x1, num_cols - 1))
          x2 = tf.maximum(x1 + 1, tf.minimum(x2, num_cols))
          
          new_shape = [input_shape[0], y2 - y1, x2 - x1, input_shape[3]]

          x_crop = x[:, :, y1:y2, x1:x2] if self.dim_ordering == 'channels_first' else x[:, y1:y2, x1:x2, :]
          xm = tf.reshape(x_crop, new_shape)
          pooled_val = tf.reduce_max(xm, axis=axis)
          outputs.append(pooled_val)

    outputs = tf.concat(outputs, axis=1)
    if self.dim_ordering == 'channels_last':
      outputs = tf.concat(outputs, axis=1)
      outputs = tf.reshape(outputs, (input_shape[0], self.num_outputs_per_channel * self.nb_channels))

    return outputs
  
  def get_config(self):
    config = {'pool_list': self.pool_list}
    base_config = super(SpatialPyramidPooling, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))