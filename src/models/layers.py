import os
from typing import Any, Literal

import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras import layers

# implementation of https://arxiv.org/pdf/1702.01381


class CNNBranch(tf.keras.layers.Layer):
  def __init__(
      self,
      branch_type: Literal['cnnA', 'cnnAspp', 'cnnBspp', 'cnnB'],
      load_weights: bool,
      train_weights: bool,
      image_width: int,
      image_height: int,
      **kwargs: Any
  ) -> None:
    super(CNNBranch, self).__init__(**kwargs)

    self.branch_type = branch_type
    self.load_weights = load_weights
    self.train_weights = train_weights

    # Determine configuration based on branch type
    self.use_final_pool = branch_type in ['cnnA', 'cnnAspp']
    self.use_spp = branch_type in ['cnnAspp', 'cnnBspp']

    # Set SPP levels based on branch type
    if branch_type == 'cnnAspp' and (image_height, image_width) == (102, 102):
      self.spp_levels = [1, 2]
    elif branch_type == 'cnnAspp':
      self.spp_levels = [1, 2, 3, 6]  # 4-level SPP
    elif branch_type == 'cnnBspp':
      self.spp_levels = [1, 2, 3, 6, 13]  # 5-level SPP
    else:
      self.spp_levels = None

    layers_list = [
        # convB1[96,11,4,0]
        layers.Conv2D(96, kernel_size=11, strides=4, padding='valid', activation='relu', name='conv1'),
        # pool[3,2]
        layers.MaxPooling2D(pool_size=3, strides=2, padding='valid', name='pool1'),
        # convB2[256,5,1,2]
        layers.Conv2D(256, kernel_size=5, strides=1, groups=2, padding='same', activation='relu', name='conv2'),
        # pool[3,2]
        layers.MaxPooling2D(pool_size=3, strides=2, padding='valid', name='pool2'),
        # convB3[384,3,1,1]
        layers.Conv2D(384, kernel_size=3, strides=1, padding='same', activation='relu', name='conv3'),
        # convB4[384,3,1,1]
        layers.Conv2D(384, kernel_size=3, strides=1, groups=2, padding='same', activation='relu', name='conv4'),
        # convB5[256,3,1,1]
        layers.Conv2D(256, kernel_size=3, strides=1, groups=2, padding='same', activation='relu', name='conv5'),
    ]

    # Add final pooling layer if needed (for cnnA and cnnAspp)
    if self.use_final_pool:
      layers_list.append(layers.MaxPooling2D(pool_size=3, strides=2, padding='valid', name='pool3'))

    # Add SPP layer if needed
    if self.use_spp:
      layers_list.append(SpatialPyramidPooling(self.spp_levels, name='spp'))
    else:
      # Flatten for fixed-size branches
      layers_list.append(layers.Flatten(name='flatten'))

    self.layers = tf.keras.Sequential(layers_list, name=f'shared_cnn_{self.branch_type}')

  def call(self, inputs):
    x = self.layers(inputs)
    return x

  def build(self, input_shape):
    # Build the sequential model
    self.layers.build(input_shape)

    # Calculate and cache output size
    self._output_size = self._calculate_output_size(input_shape)

    # Only can load the pretrained weights when have 3 channels otherwise there is a size mismatch
    if input_shape[-1] == 3 and self.load_weights:
      self._load_pretrained_weights()

    super(CNNBranch, self).build(input_shape)

  def _calculate_output_size(self, input_shape):
    """Calculate the output size from the shared CNN branch"""
    # Create a dummy input to trace through the network
    # (None, H, W, C) -> (1, H, W, C)
    concrete_shape = list(input_shape)
    concrete_shape[0] = 1  # Use batch size of 1 for calculation
    dummy_input = tf.zeros(concrete_shape)
    cnn_output = self.layers(dummy_input)
    return int(cnn_output.shape[-1])

  @property
  def output_size(self):
    """Get the output size of this CNN branch"""
    if self._output_size is None:
      raise ValueError('CNNBranch must be built before accessing output_size')
    return self._output_size

  def _load_pretrained_weights(self) -> None:
    # Set weights for each conv layer
    layer_params = self._get_layer_params()
    conv_layers = ['conv1', 'conv2', 'conv3', 'conv4', 'conv5']
    for layer_name in conv_layers:
      if layer_name in layer_params:
        layer = self.layers.get_layer(layer_name)
        layer.set_weights([
            layer_params[layer_name]['weights'],
            layer_params[layer_name]['bias']
        ])
        # Make the entire layer non-trainable
        layer.trainable = self.train_weights

  # loads weights and bias from layers in hybrid model
  def _get_layer_params(self, file_path: str = '../../pre-trained/hybrid/weights.npy') -> dict:
    # Get the directory where this function is defined
    current_dir = os.path.dirname(os.path.abspath(__file__))

    # Make the path relative to this file's directory
    absolute_path = os.path.join(current_dir, file_path)

    if not os.path.exists(absolute_path):
      raise FileNotFoundError(f'File not found: {absolute_path}')

    try:
      weights_dict = np.load(absolute_path, allow_pickle=True).item()
    except BaseException:
      weights_dict = np.load(absolute_path, allow_pickle=True, encoding='bytes').item()

    return weights_dict

# https://github.com/ShahzaibWaseem/SpatialPyramidPooling_tf2/blob/master/SpatialPyramidPooling.py


class SpatialPyramidPooling(layers.Layer):
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
    assert self.dim_ordering in {
        'channels_last',
        'channels_first',
    }, 'dim_ordering must be in {channels_last, channels_first}'

    self.pool_list = pool_list
    self.num_outputs_per_channel = sum([i * i for i in pool_list])

    super(SpatialPyramidPooling, self).__init__(**kwargs)

  def build(self, input_shape):
    if self.dim_ordering == 'channels_first':
      self.nb_channels = input_shape[1]
      h = input_shape[2]
      w = input_shape[3]
    elif self.dim_ordering == 'channels_last':
      self.nb_channels = input_shape[3]
      h = input_shape[1]
      w = input_shape[2]

    if not isinstance(h, int) or not isinstance(w, int):
      raise ValueError(
          'SpatialPyramidPooling requires static spatial dimensions, '
          f'got H={h}, W={w}'
      )

    self.pool_params = []

    for pool in self.pool_list:
      k_h = int(np.ceil(h / pool))
      k_w = int(np.ceil(w / pool))
      s_h = int(np.floor(h / pool))
      s_w = int(np.floor(w / pool))

      self.pool_params.append((k_h, k_w, s_h, s_w))

    super(SpatialPyramidPooling, self).build(input_shape)

  def compute_output_shape(self, input_shape):
    return (input_shape[0], self.nb_channels * self.num_outputs_per_channel)

  def call(self, x):
    outputs = []

    if self.dim_ordering == 'channels_first':
      for (k_h, k_w, s_h, s_w) in self.pool_params:
        pooled = tf.nn.max_pool2d(
            x,
            ksize=[1, 1, k_h, k_w],
            strides=[1, 1, s_h, s_w],
            padding='SAME',
            data_format='NCHW'
        )

        pooled = tf.reshape(
            pooled,
            [tf.shape(x)[0], -1]
        )

        outputs.append(pooled)

    elif self.dim_ordering == 'channels_last':
      for (k_h, k_w, s_h, s_w) in self.pool_params:
        pooled = tf.nn.max_pool2d(
            x,
            ksize=[1, k_h, k_w, 1],
            strides=[1, s_h, s_w, 1],
            padding='SAME',
            data_format='NHWC'
        )

        pooled = tf.reshape(
            pooled,
            [tf.shape(x)[0], -1]
        )

        outputs.append(pooled)

    return tf.concat(outputs, axis=-1)

  def get_config(self):
    config = {'pool_list': self.pool_list}
    base_config = super(SpatialPyramidPooling, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))
