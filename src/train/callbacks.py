import tensorflow as tf

def get_default_callbacks():
  """Return default training callbacks."""
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