import tensorflow as tf
from tensorflow.keras import layers

# implementation of https://arxiv.org/pdf/1702.01381
class RelativePoseModel(tf.keras.Model):
  def __init__(self, image_height=102, image_width=102, channels=1, branch_type='cnnAspp'):
    super(RelativePoseModel, self).__init__()
    
    # Store dimensions
    self.image_height = image_height
    self.image_width = image_width
    self.channels = channels
    self.branch_type = branch_type
    
    # Determine configuration based on branch type
    self.use_final_pool = branch_type in ['cnnA', 'cnnAspp']
    self.use_spp = branch_type in ['cnnAspp', 'cnnBspp']
    
    # Set SPP levels based on branch type
    if branch_type == 'cnnAspp':
      self.spp_levels = [1, 2, 3, 6]  # 4-level SPP
    elif branch_type == 'cnnBspp':
      self.spp_levels = [1, 2, 3, 6, 13]  # 5-level SPP
    else:
      self.spp_levels = None
    
    # Create shared CNN branch
    self.shared_cnn = self._create_cnn_branch()
    
    # Numerical data processor
    self.numerical_encoder = tf.keras.Sequential([
      layers.Dense(32, activation='relu'),
      layers.Dense(16, activation='relu')
    ], name='numerical_encoder')
    
    # Regression part - FC1 now needs to handle concatenated features
    self.fc1 = layers.Dense(64, activation='relu', name='FC1')  # Increased size
    self.fc2 = layers.Dense(4, name='quaternion_output')

  def _create_cnn_branch(self):
    """Create a single CNN branch with configurable ending based on branch type."""
    layers_list = [
      # convB1[96,11,4,0]
      layers.Conv2D(96, kernel_size=11, strides=4, padding='valid', 
                    activation='relu', name='convB1'),
      # pool[3,2]
      layers.MaxPooling2D(pool_size=3, strides=2, name='pool1'),
      
      # convB2[256,5,1,2]
      layers.Conv2D(256, kernel_size=5, strides=1, padding='same', 
                    activation='relu', name='convB2'),
      # pool[3,2]
      layers.MaxPooling2D(pool_size=3, strides=2, name='pool2'),
      
      # convB3[384,3,1,1]
      layers.Conv2D(384, kernel_size=3, strides=1, padding='same', 
                    activation='relu', name='convB3'),
      
      # convB4[384,3,1,1]
      layers.Conv2D(384, kernel_size=3, strides=1, padding='same', 
                    activation='relu', name='convB4'),
      
      # convB5[256,3,1,1]
      layers.Conv2D(256, kernel_size=3, strides=1, padding='same', 
                    activation='relu', name='convB5'),
    ]
    
    # Add final pooling layer if needed (for cnnA and cnnAspp)
    if self.use_final_pool:
      layers_list.append(
        layers.MaxPooling2D(pool_size=3, strides=2, padding='valid', name='pool3')
      )
    
    # Add SPP layer if needed
    if self.use_spp:
      layers_list.append(
        layers.SpatialPyramidPooling2D(self.spp_levels, name='spp')
      )
    else:
      # Flatten for fixed-size branches
      layers_list.append(layers.Flatten(name='flatten'))
    
    branch = tf.keras.Sequential(layers_list, name=f'shared_cnn_{self.branch_type}')
    
    return branch

  """
  def build(self, input_shape):
    #Build the model with proper input shapes
    # For dictionary input with 'image_data' and 'numerical' keys
    if isinstance(input_shape, dict):
      image_shape = input_shape.get('image_data', (None, 3, self.image_height, self.image_width, self.channels))
      numerical_shape = input_shape.get('numerical', (None, 4))
      
      # Extract dimensions from image_shape
      batch_size = image_shape[0]
      num_images = image_shape[1]  # Should be 3
      height = image_shape[2] if image_shape[2] is not None else self.image_height
      width = image_shape[3] if image_shape[3] is not None else self.image_width
      channels = image_shape[4]
      
      # Build shared CNN with single image shape
      single_image_shape = (batch_size, height, width, channels)
    else:
      # Default shapes
      single_image_shape = (None, self.image_height, self.image_width, self.channels)
      numerical_shape = (None, 4)
    
    # Build the shared CNN branch
    self.shared_cnn.build(single_image_shape)
    
    # Build numerical encoder
    self.numerical_encoder.build(numerical_shape)
    
    # Calculate output size based on branch type
    if self.use_spp:
      # SPP output: sum of squares of pyramid levels * number of channels
      num_channels = 256  # from convB5
      spp_output_size = sum([level * level for level in self.spp_levels]) * num_channels
      image_features_size = spp_output_size * 3
    else:
      # For fixed-size branches, calculate the flattened size
      # This depends on the input size and the network architecture
      # For 227x227 input with cnnA: output is 6x6x256 = 9216
      # For 227x227 input with cnnB: output is 13x13x256 = 43264
      if self.use_final_pool:  # cnnA
        # After all pooling layers: 227 -> 54 -> 26 -> 13 -> 6
        feature_map_size = 6 * 6 * 256
      else:  # cnnB
        # Without final pool: 227 -> 54 -> 26 -> 13
        feature_map_size = 13 * 13 * 256
      image_features_size = feature_map_size * 3
    
    # Build regression layers
    # 3 branches concatenated + numerical features
    numerical_features_size = 16  # Output of numerical_encoder
    concatenated_size = image_features_size + numerical_features_size
    
    self.fc1.build((None, concatenated_size))
    self.fc2.build((None, 64))"""
  
  def call(self, inputs):
    # Handle dictionary input
    if isinstance(inputs, dict):
      images = inputs['image_data']     # Shape: (batch_size, 3, H, W, C)
      numerical = inputs['numerical']   # Shape: (batch_size, 4)
    else:
      raise ValueError("Input must be a dictionary with 'image_data' and 'numerical' keys")
    
    # Split the 3 images
    image1 = images[:, 0, :, :, :]  # Shape: (batch_size, H, W, C)
    image2 = images[:, 1, :, :, :]
    image3 = images[:, 2, :, :, :]
    
    # Process each image through the shared CNN branch
    features1 = self.shared_cnn(image1)
    features2 = self.shared_cnn(image2)
    features3 = self.shared_cnn(image3)
    
    # Concatenate features from all three branches
    image_features = layers.Concatenate(axis=-1)([features1, features2, features3])
    
    # Process numerical input
    numerical_features = self.numerical_encoder(numerical)
    
    # Concatenate all features
    combined_features = tf.concat([image_features, numerical_features], axis=-1)
    
    # Regression part
    x = self.fc1(combined_features)
    quaternion_output = self.fc2(x)
    
    # Normalize quaternion to unit length
    quaternion_normalized = tf.nn.l2_normalize(quaternion_output, axis=-1)
    
    return quaternion_normalized