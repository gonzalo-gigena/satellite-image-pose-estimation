import numpy as np

def quaternion_to_rotation_degrees(q):
  """Convert quaternion to Euler angles in degrees."""
  w, x, y, z = q[0], q[1], q[2], q[3]
  
  # Roll (x-axis rotation)
  sinr_cosp = 2 * (w * x + y * z)
  cosr_cosp = 1 - 2 * (x * x + y * y)
  roll = np.arctan2(sinr_cosp, cosr_cosp)
  
  # Pitch (y-axis rotation)
  sinp = 2 * (w * y - z * x)
  if np.abs(sinp) >= 1:
    pitch = np.copysign(np.pi / 2, sinp)
  else:
    pitch = np.arcsin(sinp)
  
  # Yaw (z-axis rotation)
  siny_cosp = 2 * (w * z + x * y)
  cosy_cosp = 1 - 2 * (y * y + z * z)
  yaw = np.arctan2(siny_cosp, cosy_cosp)
  
  # Convert to degrees
  return np.array([roll, pitch, yaw]) * 180.0 / np.pi