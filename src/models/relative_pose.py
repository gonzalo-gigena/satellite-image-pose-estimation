import tensorflow as tf
from tensorflow.keras import layers

from models.layers import CNNBranch


class RelativePoseModel(tf.keras.Model):
  def __init__(
          self,
          image_height=102,
          image_width=102,
          channels=1,
          frames=3,
          branch_type='cnnAspp',
          load_weights=False):
    super(RelativePoseModel, self).__init__()

    # Store dimensions
    self.image_height = image_height
    self.image_width = image_width
    self.channels = channels
    self.branch_type = branch_type
    self.frames = frames

    # Create shared CNN branch
    self.shared_cnn = CNNBranch(self.branch_type, load_weights)

    # Numerical data processor
    self.numerical_encoder = tf.keras.Sequential(
        [layers.Dense(32, activation='relu'), layers.Dense(16, activation='relu')], name='numerical_encoder'
    )

    # Regression part - FC1 now needs to handle concatenated features
    self.fc1 = layers.Dense(64, activation='relu', name='FC1')  # Increased size
    self.fc2 = layers.Dense(4, name='quaternion_output')

  def call(self, inputs):
    # Handle dictionary input
    if isinstance(inputs, dict):
      images = inputs['image_data']  # Shape: (batch_size, F, H, W, C)
      numerical = inputs['numerical']  # Shape: (batch_size, 4)
    else:
      raise ValueError("Input must be a dictionary with 'image_data' and 'numerical' keys")

    # Split the images and extract features
    features = []
    for i in range(self.frames):
      image_batch = images[:, i, :, :, :]  # Shape: (batch_size, H, W, C)
      feature = self.shared_cnn(image_batch)
      features.append(feature)

    # Concatenate features from all branches
    image_features = layers.Concatenate(axis=-1)(features)

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

  def build(self, input_shape):
    """Build the model with proper input shapes"""
    # Handle dictionary input shape
    if isinstance(input_shape, dict):
      image_shape = input_shape.get(
          'image_data', (None, self.frames, self.image_height, self.image_width, self.channels)
      )
      numerical_shape = input_shape.get('numerical', (None, 4))
    else:
      # Default shapes if not dictionary
      image_shape = (None, self.frames, self.image_height, self.image_width, self.channels)
      numerical_shape = (None, 4)

    # Extract single image shape for building shared CNN
    single_image_shape = (image_shape[0], self.image_height, self.image_width, self.channels)

    # Build the shared CNN branch
    self.shared_cnn.build(single_image_shape)

    # Build numerical encoder
    self.numerical_encoder.build(numerical_shape)

    # Calculate the output size from shared CNN
    cnn_feature_size = self.shared_cnn.output_size

    # Total image features from N branches (image1, image2, ..., N)
    image_features_size = cnn_feature_size * self.frames

    # Numerical features size (output of numerical_encoder)
    numerical_features_size = 16  # Last Dense layer in numerical_encoder

    # Total concatenated size
    concatenated_size = image_features_size + numerical_features_size

    # Build regression layers
    self.fc1.build((None, concatenated_size))
    self.fc2.build((None, 64))  # FC1 output size

    # Call parent build
    super(RelativePoseModel, self).build(input_shape)
