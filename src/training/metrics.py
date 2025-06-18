from typing import List
import tensorflow as tf


from losses.custom import quaternion_loss, angular_distance_loss, detailed_distance_loss, geodesic_loss

# TODO: Add configuration
def get_metrics() -> List[tf.keras.metrics.Metric]:
  """Return default metrics.

  Returns:
    List of Keras metrics for training monitoring and optimization
  """
  return [
    'mae',
    quaternion_loss,
    angular_distance_loss,
    detailed_distance_loss,
    geodesic_loss
  ]