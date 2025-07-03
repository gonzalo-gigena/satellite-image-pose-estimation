import numpy as np
import tensorflow as tf


def quaternion_to_rotation_degrees(q):
  """Convert quaternion to Euler angles in degrees using TensorFlow operations."""
  w, x, y, z = q[0], q[1], q[2], q[3]

  # Roll (x-axis rotation)
  sinr_cosp = 2 * (w * x + y * z)
  cosr_cosp = 1 - 2 * (x * x + y * y)
  roll = tf.atan2(sinr_cosp, cosr_cosp)

  # Pitch (y-axis rotation)
  sinp = 2 * (w * y - z * x)
  # Handle gimbal lock
  pitch = tf.where(tf.abs(sinp) >= 1, tf.sign(sinp) * (tf.constant(np.pi) / 2), tf.asin(sinp))

  # Yaw (z-axis rotation)
  siny_cosp = 2 * (w * z + x * y)
  cosy_cosp = 1 - 2 * (y * y + z * z)
  yaw = tf.atan2(siny_cosp, cosy_cosp)

  # Convert to degrees
  angles_rad = tf.stack([roll, pitch, yaw])
  angles_deg = angles_rad * 180.0 / tf.constant(np.pi)

  return angles_deg
