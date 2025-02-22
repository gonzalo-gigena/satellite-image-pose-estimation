import tensorflow as tf

def get_optimizer(optimizer_name, learning_rate):
  """Select and return the optimizer based on the given name."""
  optimizers = {
    'adam': tf.keras.optimizers.Adam,
    'sgd': tf.keras.optimizers.SGD,
    'rmsprop': tf.keras.optimizers.RMSprop,
  }

  optimizer_class = optimizers.get(optimizer_name.lower())
  if optimizer_class is None:
    raise ValueError(f"Unsupported optimizer: {optimizer_name}")

  return optimizer_class(learning_rate=learning_rate)