import tensorflow as tf
import numpy as np

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