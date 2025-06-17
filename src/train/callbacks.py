from typing import List
import tensorflow as tf

# TODO: Add configuration
def get_default_callbacks() -> List[tf.keras.callbacks.Callback]:
  """Return default training callbacks.

  Returns:
    List of Keras callbacks for training monitoring and optimization
  """
  return [
    tf.keras.callbacks.EarlyStopping(
      monitor='val_mae',
      patience=10,
      restore_best_weights=True
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
      monitor='val_mae',
      factor=0.5,
      patience=5,
      min_lr=1e-6
    )
  ]