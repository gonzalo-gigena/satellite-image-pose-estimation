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

class DataGenerator(tf.keras.utils.Sequence):
  def __init__(self, images, numerical, targets, shuffle, batch_size=32):
    self.images = tf.convert_to_tensor(images, dtype=tf.float32)
    self.numerical = tf.convert_to_tensor(numerical, dtype=tf.float32)
    self.targets = tf.convert_to_tensor(targets, dtype=tf.float32)
    self.batch_size = batch_size
    self.indexes = np.arange(len(self.targets))
    self.shuffle = shuffle

    # If shuffle is True, shuffle the indices right away
    if self.shuffle:
      np.random.shuffle(self.indexes)

  def on_epoch_end(self):
    """Called at the end of every epoch"""
    if self.shuffle:
      np.random.shuffle(self.indexes)

  def __len__(self):
    return int(np.ceil(len(self.images) / self.batch_size))

  def __getitem__(self, idx):
    batch_slice = slice(idx * self.batch_size, (idx + 1) * self.batch_size)

    batch_images = self.images[batch_slice]
    batch_numerical = self.numerical[batch_slice]
    batch_targets = self.targets[batch_slice]

    return {
      'image_data': batch_images,
      'numerical': batch_numerical
    }, batch_targets