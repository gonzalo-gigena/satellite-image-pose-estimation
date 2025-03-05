import os
import numpy as np
import tensorflow as tf

from .light_glue_model import extract_points
from data_generator import DataGenerator
from data_loader import DataLoader


class MatchingDataGenerator(DataGenerator):
  def __init__(self, points, numerical, targets, shuffle, batch_size, augment):
    self.data = tf.ragged.constant(points).to_tensor()
    super().__init__(numerical, targets, shuffle, batch_size, augment)

class MatchingDataLoader(DataLoader):
  def __init__(self, data_path, train_split, validation_split, seed, matching_method, num_matches):
    assert matching_method is not None, "matching_method must be specified for MatchingDataLoader"
    assert num_matches is not None, "num_matches must be specified for MatchingDataLoader"
    super().__init__(data_path, train_split, validation_split, seed)
    self.matching_method = matching_method
    self.num_matches = num_matches

  def _process_data(self, files):
    """
    Process data by matching features between consecutive images

    Args:
      files: List of filenames in the data directory

    Returns:
      Processed and split data for training
    """
    points = []
    numerical_data = []
    targets = []

    for i in range(len(files)-1)[:10]:
      print(f"Processing image pair {i + 1}/{len(files) - 1}")
      filename0 = files[i]
      filename1 = files[i + 1]

      timestamp0, sat_pos0, sat_rot0 = self._extract_data_from_filename(filename0)
      timestamp1, sat_pos1, sat_rot1 = self._extract_data_from_filename(filename1)

      # coordinates with shape (K, 2)
      points0, points1 = self._matching_features(filename0, filename1)

      # Get the number of matches K
      K = points0.shape[0]

      # Check if there are no matches
      #if K == 0:
      #  print(f"No matches found between images {i} and {i + 1}")
      #  continue

      if K >= self.num_matches:
        # Keep only the first N matches
        points0 = points0[:self.num_matches]
        points1 = points1[:self.num_matches]
      else:
        # Pad with dummy matches if there are fewer than N matches
        num_missing = self.num_matches - K
        # Create dummy matches with zeros
        dummy_points0 = tf.zeros((num_missing, 2), dtype=tf.float32)
        dummy_points1 = tf.zeros((num_missing, 2), dtype=tf.float32)
        # Concatenate the real and dummy matches
        points0 = tf.concat([points0.cpu().numpy(), dummy_points0], axis=0)
        points1 = tf.concat([points1.cpu().numpy(), dummy_points1], axis=0)

      # This will result in shape (2N, 2)
      points.append(np.concatenate((
        points0.numpy().reshape(-1),
        points1.numpy().reshape(-1)
      )))

      # Concatenate timestamp and satellite position
      numerical_data.append(
        np.concatenate([
          np.array([timestamp0]),  # Convert scalar to 1D array
          np.array([timestamp1]),
          sat_pos0,
          sat_pos1
        ], axis=0)
      )
      # Satellite rotation is the target, already normalized quaternion
      targets.append(np.concatenate([sat_rot0, sat_rot1], axis=0))

    # Convert lists to numpy arrays
    numerical_data = np.array(numerical_data)
    targets = np.array(targets)
    points = np.array(points)

    # Split data
    return self._split_data(points, numerical_data, targets)

  def _matching_features(self, img0, img1):
    """
    Perform feature matching between two images

    Args:
      img0: Filename of the first image
      img1: Filename of the second image

    Returns:
      points0: Keypoints from the first image
      points1: Corresponding keypoints from the second image
    """
    image0_path = os.path.join(self.data_path, img0)
    image1_path = os.path.join(self.data_path, img1)

    # Implement your matching method here
    # For example, using LightGlue or any other method
    if self.matching_method == 'light_glue':
      return extract_points(image0_path, image1_path)
    else:
      raise NotImplementedError(f"Matching method '{self.matching_method}' is not implemented")

class FeatureMatchingModel(tf.keras.Model):
  def __init__(self):
    super(FeatureMatchingModel, self).__init__()
    
    # Feature processing layers
    self.feature_encoder = tf.keras.Sequential([
      tf.keras.layers.Dense(128, activation='relu'),
      tf.keras.layers.Dense(64, activation='relu')
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
      tf.keras.layers.Dense(8)  # 4 quaternion components for each image
    ])

  def reshape_points(self, points, k):
    # Reshape the 1D array into pairs of coordinates
    return tf.reshape(points, (-1, k, 2))

  def call(self, inputs):
    # Unpack inputs
    points = inputs['image_data']  # shape: (batch_size, 2K)
    timestamp_pos = inputs['numerical']  # shape: (batch_size, 8)
    
    # Calculate K (number of points) dynamically
    k = tf.shape(points)[1] // 2
    
    # Reshape points into two sets of K points
    points0 = self.reshape_points(points[:, :k], k//2)  # First half of points
    points1 = self.reshape_points(points[:, k:], k//2)  # Second half of points
    
    # Process matched features
    features0 = self.feature_encoder(points0)  # shape: (batch_size, K/2, 64)
    features1 = self.feature_encoder(points1)  # shape: (batch_size, K/2, 64)
    
    # Global average pooling to get fixed-size representations
    global_features0 = tf.reduce_mean(features0, axis=1)
    global_features1 = tf.reduce_mean(features1, axis=1)
    
    # Process position and timestamp information
    pos_features = self.position_encoder(timestamp_pos)
    
    # Concatenate all features
    combined_features = tf.concat([
      global_features0,
      global_features1,
      pos_features
    ], axis=-1)
    
    # Predict quaternions
    quaternions = self.quaternion_predictor(combined_features)
    
    # Normalize the quaternions
    q0, q1 = tf.split(quaternions, 2, axis=-1)
    q0_normalized = tf.math.l2_normalize(q0, axis=-1)
    q1_normalized = tf.math.l2_normalize(q1, axis=-1)
    
    return tf.concat([q0_normalized, q1_normalized], axis=-1)