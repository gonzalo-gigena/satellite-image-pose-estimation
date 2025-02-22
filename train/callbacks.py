import tensorflow as tf

def get_default_callbacks():
  """Return default training callbacks."""
  return [
    tf.keras.callbacks.ReduceLROnPlateau(
      monitor='val_loss',
      factor=0.5,
      patience=5,
      min_lr=1e-6
    ),
    tf.keras.callbacks.EarlyStopping(
      monitor='val_loss',
      patience=15,
      restore_best_weights=True
    )
  ]