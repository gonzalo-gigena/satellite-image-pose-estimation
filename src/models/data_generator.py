import tensorflow as tf
import numpy as np

class DataGenerator(tf.keras.utils.Sequence):
  def __init__(self, numerical, targets, shuffle=False, batch_size=32, augment=False):
    self.numerical = tf.convert_to_tensor(numerical, dtype=tf.float32)
    self.targets = tf.convert_to_tensor(targets, dtype=tf.float32)
    self.batch_size = batch_size
    self.indexes = np.arange(len(self.targets))
    self.shuffle = shuffle

    # If shuffle is True, shuffle the indices right away
    if self.shuffle:
      np.random.shuffle(self.indexes)
    
    self.augment = augment
    self.augmentation  = tf.keras.Sequential([
      # Brightness variation (simulate lighting conditions)
      tf.keras.layers.RandomBrightness(factor=0.2),
      
      # Contrast variation (simulate atmospheric effects)
      tf.keras.layers.RandomContrast(factor=0.2),
  
      # Gaussian noise (simulate sensor noise)
      tf.keras.layers.GaussianNoise(0.1)
    ])

  def on_epoch_end(self):
    """Called at the end of every epoch"""
    if self.shuffle:
      np.random.shuffle(self.indexes)

  def __len__(self):
    return int(np.ceil(len(self.data) / self.batch_size))

  def __getitem__(self, idx):
    batch_slice = slice(idx * self.batch_size, (idx + 1) * self.batch_size)

    batch_data = self.data[batch_slice]
    batch_numerical = self.numerical[batch_slice]
    batch_targets = self.targets[batch_slice]
    
    if self.augment:
      batch_data = self.augmentation(batch_data, training=True)

    return {
      'image_data': batch_data,
      'numerical': batch_numerical
    }, batch_targets