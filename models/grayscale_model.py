import tensorflow as tf
import numpy as np

class GrayscaleModel(tf.keras.Model):
  def __init__(self, image_height=102, image_width=102):
    super(GrayscaleModel, self).__init__()

    # Image processing layers
    self.image_encoder = tf.keras.Sequential([
      tf.keras.layers.Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(image_height, image_width, 1)),
      tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
      tf.keras.layers.Conv2D(64, kernel_size=(3, 3), activation='relu'),
      tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
      tf.keras.layers.Conv2D(128, kernel_size=(3, 3), activation='relu'),
      tf.keras.layers.GlobalAveragePooling2D()
    ])

    # Position and timestamp processing
    self.position_encoder = tf.keras.Sequential([
      tf.keras.layers.Dense(64, activation='relu'),
      tf.keras.layers.Dense(32, activation='relu')
    ])

    # Final prediction layers
    self.quaternion_predictor = tf.keras.Sequential([
      tf.keras.layers.Dense(256, activation='relu'),
      tf.keras.layers.Dense(128, activation='relu'),
      tf.keras.layers.Dense(4)  # 4 quaternion components
    ])

  def call(self, inputs):
    # Unpack inputs
    images = inputs['image_data']                # Shape: (batch_size, height, width, 1)
    numerical = inputs['numerical']          # Shape: (batch_size, numerical_features)

    # Process images through CNN
    image_features = self.image_encoder(images)  # Shape: (batch_size, feature_dim)

    # Process numerical data
    numerical_features = self.position_encoder(numerical)  # Shape: (batch_size, 32)

    # Concatenate all features
    combined_features = tf.concat([image_features, numerical_features], axis=-1)

    # Predict quaternion
    quaternions = self.quaternion_predictor(combined_features)  # Shape: (batch_size, 4)

    # Normalize the quaternion
    quaternions_normalized = tf.math.l2_normalize(quaternions, axis=-1)

    return quaternions_normalized

class ImprovedGrayscaleModel(tf.keras.Model):
  def __init__(self, image_height=102, image_width=102):
    super(ImprovedGrayscaleModel, self).__init__()
    
    # Improved Image Encoder with ResNet blocks
    def residual_block(x, filters, kernel_size=3):
      shortcut = x
      x = tf.keras.layers.Conv2D(filters, kernel_size, padding='same')(x)
      x = tf.keras.layers.BatchNormalization()(x)
      x = tf.keras.layers.ReLU()(x)
      x = tf.keras.layers.Conv2D(filters, kernel_size, padding='same')(x)
      x = tf.keras.layers.BatchNormalization()(x)
      if shortcut.shape[-1] != filters:
        shortcut = tf.keras.layers.Conv2D(filters, 1, padding='same')(shortcut)
      x = tf.keras.layers.Add()([shortcut, x])
      return tf.keras.layers.ReLU()(x)

    # Image processing layers
    inputs = tf.keras.layers.Input(shape=(image_height, image_width, 1))
    x = tf.keras.layers.Conv2D(64, 7, strides=2, padding='same')(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.MaxPooling2D(3, strides=2, padding='same')(x)
    
    # ResNet blocks
    x = residual_block(x, 64)
    x = residual_block(x, 64)
    x = residual_block(x, 128, kernel_size=3)
    x = residual_block(x, 128, kernel_size=3)
    x = residual_block(x, 256, kernel_size=3)
    
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    self.image_encoder = tf.keras.Model(inputs=inputs, outputs=x)

    # Improved position encoder
    self.position_encoder = tf.keras.Sequential([
      tf.keras.layers.Dense(128, activation='relu'),
      tf.keras.layers.BatchNormalization(),
      tf.keras.layers.Dropout(0.3),
      tf.keras.layers.Dense(64, activation='relu'),
      tf.keras.layers.BatchNormalization(),
    ])

    # Improved quaternion predictor
    self.quaternion_predictor = tf.keras.Sequential([
      tf.keras.layers.Dense(512, activation='relu'),
      tf.keras.layers.BatchNormalization(),
      tf.keras.layers.Dropout(0.3),
      tf.keras.layers.Dense(256, activation='relu'),
      tf.keras.layers.BatchNormalization(),
      tf.keras.layers.Dense(128, activation='relu'),
      tf.keras.layers.Dense(4)
    ])

  def call(self, inputs, training=False):
    images = inputs['image_data']
    numerical = inputs['numerical']

    image_features = self.image_encoder(images, training=training)
    numerical_features = self.position_encoder(numerical, training=training)
    
    combined_features = tf.concat([image_features, numerical_features], axis=-1)
    quaternions = self.quaternion_predictor(combined_features, training=training)
    return tf.math.l2_normalize(quaternions, axis=-1)

class GrayscaleModel2(tf.keras.Model):
  def __init__(self, image_height=102, image_width=102):
    super(GrayscaleModel, self).__init__()

    # Image processing with ResNet-like blocks
    self.image_encoder = tf.keras.Sequential([
      # Initial Conv
      tf.keras.layers.Conv2D(32, kernel_size=(3, 3), strides=(2, 2), padding='same',
                            input_shape=(image_height, image_width, 1)),
      tf.keras.layers.BatchNormalization(),
      tf.keras.layers.Activation('relu'),
      tf.keras.layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same'),

      # Residual blocks
      self._make_res_block(64, 2),
      self._make_res_block(128, 2),
      
      tf.keras.layers.GlobalAveragePooling2D(),
      tf.keras.layers.Dropout(0.4)
    ])

    # Position and timestamp processing
    self.position_encoder = tf.keras.Sequential([
      # Separate branches for position and timestamp
      tf.keras.layers.Dense(64, activation='relu'),
      tf.keras.layers.BatchNormalization(),
      tf.keras.layers.Dense(32, activation='relu'),
      tf.keras.layers.BatchNormalization(),
    ])

    # Timestamp specific processing
    self.time_encoder = tf.keras.Sequential([
      tf.keras.layers.Dense(32, activation='relu'),
      tf.keras.layers.BatchNormalization(),
    ])

    # Final prediction layers
    self.quaternion_predictor = tf.keras.Sequential([
      tf.keras.layers.Dense(256, activation='relu'),
      tf.keras.layers.BatchNormalization(),
      tf.keras.layers.Dropout(0.3),
      tf.keras.layers.Dense(128, activation='relu'),
      tf.keras.layers.BatchNormalization(),
      tf.keras.layers.Dense(4)  # 4 quaternion components
    ])

  def _make_res_block(self, filters, blocks):
    layers = []
    # First block with potential downsampling
    layers.append(tf.keras.layers.Conv2D(filters, kernel_size=(3, 3), 
                                        padding='same', strides=(2, 2)))
    layers.append(tf.keras.layers.BatchNormalization())
    layers.append(tf.keras.layers.Activation('relu'))
    
    # Additional blocks
    for _ in range(blocks-1):
      layers.append(tf.keras.layers.Conv2D(filters, kernel_size=(3, 3), 
                                          padding='same'))
      layers.append(tf.keras.layers.BatchNormalization())
      layers.append(tf.keras.layers.Activation('relu'))
    
    return tf.keras.Sequential(layers)

  def call(self, inputs):
    images = inputs['image_data']
    timestamp = inputs['numerical'][:, :1]  # Assuming first value is the timestamp
    position = inputs['numerical'][:, 1:]  # Assuming remaining values are position

    # Process each input type
    image_features = self.image_encoder(images)
    position_features = self.position_encoder(position)
    time_features = self.time_encoder(timestamp)

    # Combine all features
    combined_features = tf.concat([image_features, position_features, time_features], axis=-1)

    # Predict quaternion
    quaternions = self.quaternion_predictor(combined_features)
    return tf.math.l2_normalize(quaternions, axis=-1)