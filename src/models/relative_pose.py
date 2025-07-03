import tensorflow as tf
from tensorflow.keras import layers

from models.layers import SpatialPyramidPooling


# implementation of https://arxiv.org/pdf/1702.01381
class RelativePoseModel(tf.keras.Model):
  def __init__(self, image_height=102, image_width=102, channels=1, frames=3, branch_type="cnnAspp"):
    super(RelativePoseModel, self).__init__()

    # Store dimensions
    self.image_height = image_height
    self.image_width = image_width
    self.channels = channels
    self.branch_type = branch_type
    self.frames = frames

    # Determine configuration based on branch type
    self.use_final_pool = branch_type in ["cnnA", "cnnAspp"]
    self.use_spp = branch_type in ["cnnAspp", "cnnBspp"]

    # Set SPP levels based on branch type
    if branch_type == "cnnAspp":
      self.spp_levels = [1, 2, 3, 6]  # 4-level SPP
    elif branch_type == "cnnBspp":
      self.spp_levels = [1, 2, 3, 6, 13]  # 5-level SPP
    else:
      self.spp_levels = None

    # Create shared CNN branch
    self.shared_cnn = self._create_cnn_branch()

    # Numerical data processor
    self.numerical_encoder = tf.keras.Sequential(
        [layers.Dense(32, activation="relu"), layers.Dense(16, activation="relu")], name="numerical_encoder"
    )

    # Regression part - FC1 now needs to handle concatenated features
    self.fc1 = layers.Dense(64, activation="relu", name="FC1")  # Increased size
    self.fc2 = layers.Dense(4, name="quaternion_output")

  def _create_cnn_branch(self):
    """Create a single CNN branch with configurable ending based on branch type."""
    layers_list = [
        # convB1[96,11,4,0]
        layers.Conv2D(96, kernel_size=11, strides=4, padding="valid", activation="relu", name="convB1"),
        # pool[3,2]
        layers.MaxPooling2D(pool_size=3, strides=2, name="pool1"),
        # convB2[256,5,1,2]
        layers.Conv2D(256, kernel_size=5, strides=1, padding="same", activation="relu", name="convB2"),
        # pool[3,2]
        layers.MaxPooling2D(pool_size=3, strides=2, name="pool2"),
        # convB3[384,3,1,1]
        layers.Conv2D(384, kernel_size=3, strides=1, padding="same", activation="relu", name="convB3"),
        # convB4[384,3,1,1]
        layers.Conv2D(384, kernel_size=3, strides=1, padding="same", activation="relu", name="convB4"),
        # convB5[256,3,1,1]
        layers.Conv2D(256, kernel_size=3, strides=1, padding="same", activation="relu", name="convB5"),
    ]

    # Add final pooling layer if needed (for cnnA and cnnAspp)
    if self.use_final_pool:
      layers_list.append(layers.MaxPooling2D(pool_size=3, strides=2, padding="valid", name="pool3"))

    # Add SPP layer if needed
    if self.use_spp:
      layers_list.append(SpatialPyramidPooling(self.spp_levels, name="spp"))
    else:
      # Flatten for fixed-size branches
      layers_list.append(layers.Flatten(name="flatten"))

    branch = tf.keras.Sequential(layers_list, name=f"shared_cnn_{self.branch_type}")

    return branch

  def call(self, inputs):
    # Handle dictionary input
    if isinstance(inputs, dict):
      images = inputs["image_data"]  # Shape: (batch_size, F, H, W, C)
      numerical = inputs["numerical"]  # Shape: (batch_size, 4)
    else:
      raise ValueError("Input must be a dictionary with 'image_data' and 'numerical' keys")

    # Split the images and extract features
    features = []
    for i in range(self.frames):
      image = images[:, i, :, :, :]  # Shape: (batch_size, H, W, C)
      feature = self.shared_cnn(image)
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
          "image_data", (None, self.frames, self.image_height, self.image_width, self.channels)
      )
      numerical_shape = input_shape.get("numerical", (None, 4))
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
    cnn_feature_size = self._calculate_cnn_output_size(single_image_shape)

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

  def _calculate_cnn_output_size(self, input_shape):
    """Calculate the output size from the shared CNN branch"""
    # Create a dummy input to trace through the network
    # (None, H, W, C) -> (1, H, W, C)
    concrete_shape = list(input_shape)
    concrete_shape[0] = 1  # Use batch size of 1 for calculation
    dummy_input = tf.zeros(concrete_shape)
    cnn_output = self.shared_cnn(dummy_input)
    return int(cnn_output.shape[-1])
