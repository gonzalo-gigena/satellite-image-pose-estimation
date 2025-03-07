import tensorflow as tf
import numpy as np

#from data_generator import DataGenerator
from data_loader import DataLoader

class TimelessDataLoader(DataLoader):
  def _process_data(self, files):
    """
    Process data by converting images to grayscale and extracting pixels

    Args:
      files: List of filenames in the data directory

    Returns:
      Processed and split data for training
    """
    images = []
    positions = []
    targets = []

    for i in range(len(files)):
      print(f"Processing image {i + 1}/{len(files)}")
      filename = files[i]

      # Extract data from filename
      _, sat_pos, sat_rot = self._extract_data_from_filename(filename)

      # Load image and get grayscale pixels
      pixels = self._get_pixels(filename)

      # Append data to lists
      images.append(pixels)
      positions.append(sat_pos)
      targets.append(sat_rot)

    # Convert Python lists to NumPy arrays
    images = np.array(images)        # shape: (N, H, W, C)
    positions = np.array(positions)             # shape: (N, 3)
    targets = np.array(targets)               # shape: (N, 4)

    return self._split_data(images, positions, targets)

class TimelessModel(tf.keras.Model):
  def __init__(self, image_height=102, image_width=102):
    super(TimelessModel, self).__init__()

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