import tensorflow as tf


class GrayscaleModel(tf.keras.Model):
  def __init__(self, image_height=102, image_width=102, channels=1):
    super(GrayscaleModel, self).__init__()

    # Store dimensions for build method
    self.image_height = image_height
    self.image_width = image_width
    self.channels = channels

    # Image processing layers for sequential frames
    self.image_encoder = tf.keras.Sequential(
        [
            tf.keras.layers.TimeDistributed(
                tf.keras.layers.Conv2D(
                    32, kernel_size=(3, 3), activation="relu", input_shape=(image_height, image_width, channels)
                )
            ),
            tf.keras.layers.TimeDistributed(tf.keras.layers.MaxPooling2D(pool_size=(2, 2))),
            tf.keras.layers.TimeDistributed(tf.keras.layers.Conv2D(64, kernel_size=(3, 3), activation="relu")),
            tf.keras.layers.TimeDistributed(tf.keras.layers.MaxPooling2D(pool_size=(2, 2))),
            tf.keras.layers.TimeDistributed(tf.keras.layers.Conv2D(128, kernel_size=(3, 3), activation="relu")),
            tf.keras.layers.TimeDistributed(tf.keras.layers.GlobalAveragePooling2D()),
            tf.keras.layers.Flatten(),  # Flatten the sequence of features
        ]
    )

    # Single numerical data processor (for shape (batch_size, 4))
    self.numerical_encoder = tf.keras.Sequential(
        [tf.keras.layers.Dense(32, activation="relu"), tf.keras.layers.Dense(16, activation="relu")]
    )

    # Final prediction layers
    self.quaternion_predictor = tf.keras.Sequential(
        [
            tf.keras.layers.Dense(256, activation="relu"),
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.Dense(4),  # 4 quaternion components
        ]
    )

  def build(self, input_shape):
    """Build the model with proper input shapes."""

    # Create dummy inputs to build the model
    if isinstance(input_shape, list) and len(input_shape) == 2:
      # Two inputs: image_data and numerical
      image_shape, numerical_shape = input_shape
    else:
      # Single input or other format
      image_shape = (None, 3, self.image_height, self.image_width, self.channels)
      numerical_shape = (None, 4)

    # Build each component
    # For TimeDistributed layers, we need to build with the full sequential shape
    self.image_encoder.build(image_shape)
    self.numerical_encoder.build(numerical_shape)

    # Calculate the output shape of image_encoder for quaternion_predictor
    # The image_encoder outputs (batch_size, 384) after flattening
    # 3 frames * 128 features per frame = 384
    image_features_shape = (None, 384)  # 3 * 128 = 384 (3 frames * 128 features)
    numerical_features_shape = (None, 16)
    combined_shape = (None, 400)  # 384 + 16

    self.quaternion_predictor.build(combined_shape)

  def call(self, inputs):
    # Unpack inputs
    images = inputs["image_data"]  # Shape: (batch_size, 3, 102, 102, 1)
    numerical = inputs["numerical"]  # Shape: (batch_size, 4)

    # Process sequential images through CNN
    image_features = self.image_encoder(images)  # Shape: (batch_size, 384)

    # Process numerical input
    numerical_features = self.numerical_encoder(numerical)  # Shape: (batch_size, 16)

    # Concatenate all features
    combined_features = tf.concat([image_features, numerical_features], axis=-1)
    # Shape: (batch_size, 384 + 16 = 400)

    # Predict quaternion
    quaternions = self.quaternion_predictor(combined_features)  # Shape: (batch_size, 4)

    # Normalize the quaternion
    quaternions_normalized = tf.math.l2_normalize(quaternions, axis=-1)

    return quaternions_normalized
