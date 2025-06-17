import tensorflow as tf
import numpy as np
from typing import Dict, List
from numpy.typing import NDArray
from sklearn.preprocessing import StandardScaler

from .data_generator import DataGenerator
from .data_loader import DataLoader, FileMetadata


class GrayscaleDataGenerator(DataGenerator):
  pass


class GrayscaleDataLoader(DataLoader):
  def _process_data(self, files: List[str]) -> Dict[str, Dict[str, NDArray[np.floating]]]:
    """
    Process data by converting images to grayscale and extracting pixels

    Args:
      files: List of filenames in the data directory

    Returns:
      Processed and split data for training
    """
    images: List[List[NDArray[np.floating]]] = []
    time: List[str] = []
    positions: List[NDArray[np.floating]] = []
    targets: List[NDArray[np.floating]] = []
    
    for i in range(0, len(files), self.burst):
      print(f"Processing images [{i+1}-{i+self.burst}]/{len(files)}")

      files_data: List[FileMetadata] = []
      for j in range(self.burst):
        files_data.append(self._extract_data_from_filename(files[i+j]))
      
      if not self._validate(files_data):
        print("Skip sequence")
        continue

      # Load image and get grayscale pixels
      pixels: List[NDArray[np.floating]] = []
      for j in range(self.burst):
        pixels.append(self._get_pixels(files[i+j]))

      # Append data to lists
      images.append(pixels)
      time.append(files_data[-1][-3]) # time elapsed of last image
      positions.append(files_data[-1][-2]) # sat position of last image
      targets.append(files_data[-1][-1]) # sat rotation of last image

    # Convert Python lists to NumPy arrays
    images_array: NDArray[np.floating] = np.array(images)        # shape: (N, B, H, W, C)
    time_array: NDArray[np.floating] = np.array(time, dtype=np.float32)  # shape: (N,)
    positions_array: NDArray[np.floating] = np.array(positions)             # shape: (N, 3)
    targets_array: NDArray[np.floating] = np.array(targets)               # shape: (N, 4)

    #pos_scaler: StandardScaler = StandardScaler()
    #pos_data_norm: NDArray[np.floating] = pos_scaler.fit_transform(positions_array)
    numerical_data: NDArray[np.floating] = np.column_stack([time_array, positions_array])

    # (N, 3, 102, 102, 1) (N, 4) (N, 4)
    return self._split_data(images_array, numerical_data, targets_array)
    
  def _validate(self, data: List[FileMetadata]) -> bool:
    """
    Validate that all files in a sequence belong to the same position and burst.
    
    Args:
      data: List of extracted file data tuples
      
    Returns:
      True if validation passes, False otherwise
    """
    for i in range(len(data)-1):
      if data[i][0] != data[i+1][0]: 
        return False  # Make sure images are in the same position
      if data[i][1] != data[i+1][1]: 
        return False  # Make sure images are in the same burst
    return True

class GrayscaleModel(tf.keras.Model):
  def __init__(self, image_height=102, image_width=102, channels=1):
    super(GrayscaleModel, self).__init__()

    # Image processing layers for sequential frames
    self.image_encoder = tf.keras.Sequential([
      tf.keras.layers.TimeDistributed(
        tf.keras.layers.Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(image_height, image_width, channels))
      ),
      tf.keras.layers.TimeDistributed(tf.keras.layers.MaxPooling2D(pool_size=(2, 2))),
      tf.keras.layers.TimeDistributed(tf.keras.layers.Conv2D(64, kernel_size=(3, 3), activation='relu')),
      tf.keras.layers.TimeDistributed(tf.keras.layers.MaxPooling2D(pool_size=(2, 2))),
      tf.keras.layers.TimeDistributed(tf.keras.layers.Conv2D(128, kernel_size=(3, 3), activation='relu')),
      tf.keras.layers.TimeDistributed(tf.keras.layers.GlobalAveragePooling2D()),
      tf.keras.layers.Flatten()  # Flatten the sequence of features
    ])

    # Single numerical data processor (for shape (batch_size, 4))
    self.numerical_encoder = tf.keras.Sequential([
      tf.keras.layers.Dense(32, activation='relu'),
      tf.keras.layers.Dense(16, activation='relu')
    ])

    # Final prediction layers
    self.quaternion_predictor = tf.keras.Sequential([
      tf.keras.layers.Dense(256, activation='relu'),
      tf.keras.layers.Dense(128, activation='relu'),
      tf.keras.layers.Dense(4)  # 4 quaternion components
    ])

  def call(self, inputs):
    # Unpack inputs
    images = inputs['image_data']     # Shape: (batch_size, 3, 102, 102, 1)
    numerical = inputs['numerical']   # Shape: (batch_size, 4)

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