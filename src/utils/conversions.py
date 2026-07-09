import numpy as np


def quaternion_to_rotation_matrix(q: np.ndarray) -> np.ndarray:
  """Convert a unit quaternion (qw, qx, qy, qz) to a 3x3 rotation matrix.

  Args:
    q: Array of shape (4,) with components (qw, qx, qy, qz).

  Returns:
    Rotation matrix of shape (3, 3).
  """
  q = q / np.linalg.norm(q)
  qw, qx, qy, qz = q

  R = np.array([
      [1 - 2 * (qy**2 + qz**2), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
      [2 * (qx * qy + qz * qw), 1 - 2 * (qx**2 + qz**2), 2 * (qy * qz - qx * qw)],
      [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx**2 + qy**2)],
  ])
  return R


def angular_error_deg(q_pred: np.ndarray, q_true: np.ndarray) -> float:
  """Geodesic angular error in degrees between two unit quaternions.

  Handles the double-cover ambiguity (q and -q same rotation) via abs dot product.

  Args:
    q_pred: Predicted quaternion (4,).
    q_true: Ground truth quaternion (4,).

  Returns:
    Angular error in degrees.
  """
  dot = np.clip(np.abs(np.dot(q_pred, q_true)), 0.0, 1.0)
  return float(np.degrees(2.0 * np.arccos(dot)))
