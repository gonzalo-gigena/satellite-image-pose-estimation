from typing import Dict, Tuple

import numpy as np
import tensorflow as tf
from numpy.typing import NDArray


class BaseDataGenerator(tf.keras.utils.Sequence):
  """Data generator for training with image and numerical data."""

  def __init__(
      self,
      images: NDArray[np.floating],
      numerical: NDArray[np.floating],
      targets: NDArray[np.floating],
      shuffle: bool = False,
      batch_size: int = 32,
      augment: bool = False,
      **kwargs
  ) -> None:
    """
    Initialize the data generator.

    Args:
      images: Image data array
      numerical: Numerical features array
      targets: Target values array
      shuffle: Whether to shuffle data between epochs
      batch_size: Size of each batch
      augment: Whether to apply data augmentation
      **kwargs: Additional arguments for PyDataset compatibility
    """
    super().__init__(**kwargs)
    self.images = tf.convert_to_tensor(images, dtype=tf.float32)
    self.numerical = tf.convert_to_tensor(numerical, dtype=tf.float32)
    self.targets = tf.convert_to_tensor(targets, dtype=tf.float32)
    self.batch_size = batch_size
    self.indexes: NDArray[np.int_] = np.arange(len(self.targets))
    self.shuffle = shuffle

    # If shuffle is True, shuffle the indices right away
    if self.shuffle:
      np.random.shuffle(self.indexes)

    self.augment = augment
    self.augmentation: tf.keras.Sequential = tf.keras.Sequential(
        [
            # Brightness variation (simulate lighting conditions)
            tf.keras.layers.RandomBrightness(factor=0.2),
            # Contrast variation (simulate atmospheric effects)
            tf.keras.layers.RandomContrast(factor=0.2),
            # Gaussian noise (simulate sensor noise)
            tf.keras.layers.GaussianNoise(0.1),
        ]
    )

  def on_epoch_end(self) -> None:
    """Called at the end of every epoch."""
    if self.shuffle:
      np.random.shuffle(self.indexes)

  def __len__(self) -> int:
    """Return the number of batches per epoch."""
    return int(np.ceil(len(self.targets) / self.batch_size))

  def __getitem__(self, idx: int) -> Tuple[Dict[str, tf.Tensor], tf.Tensor]:
    """
    Generate one batch of data.

    Args:
      idx: Batch index

    Returns:
      Tuple of (input_dict, targets) where input_dict contains 'image_data' and 'numerical'
    """
    # Get batch indices
    batch_indices = self.indexes[idx * self.batch_size: (idx + 1) * self.batch_size]

    # Get batch data using indices
    batch_images = tf.gather(self.images, batch_indices)
    batch_numerical = tf.gather(self.numerical, batch_indices)
    batch_targets = tf.gather(self.targets, batch_indices)

    if self.augment:
      batch_images = self.augmentation(batch_images, training=True)

    return {'image_data': batch_images, 'numerical': batch_numerical}, batch_targets


class DataGenerator(BaseDataGenerator):
  pass
