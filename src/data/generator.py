from typing import Dict, List, Tuple

from data.loader import DataSplit

import numpy as np
import tensorflow as tf
from numpy.typing import NDArray


class BaseDataGenerator(tf.keras.utils.Sequence):
  """Data generator for training with image and numerical data."""

  def __init__(
      self,
      data: DataSplit,
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
    # Keep as numpy for faster indexing
    self.images = np.ascontiguousarray(data.images, dtype=np.float32)
    self.numerical = np.ascontiguousarray(data.numerical, dtype=np.float32)
    self.targets = np.ascontiguousarray(data.targets, dtype=np.float32)
    self.batch_size = batch_size
    self.shuffle = shuffle
    self.augment = augment
    self.n_samples = len(self.targets)

    # Pre-compute batch indices
    self.indexes: NDArray[np.int_] = np.arange(self.n_samples)
    self._batch_indices: List[NDArray[np.int_]] = []
    self._compute_batch_indices()

    # Initialize augmentation layer once
    if self.augment:
      self.augmentation: tf.keras.Sequential = tf.keras.Sequential(
          [
              tf.keras.layers.RandomBrightness(factor=0.2),
              tf.keras.layers.RandomContrast(factor=0.2),
              tf.keras.layers.GaussianNoise(0.1),
          ]
      )

  def _compute_batch_indices(self) -> None:
    """Pre-compute batch indices for all batches."""
    if self.shuffle:
      np.random.shuffle(self.indexes)

    self._batch_indices = [
        self.indexes[i * self.batch_size:(i + 1) * self.batch_size]
        for i in range(len(self))
    ]

  def on_epoch_end(self) -> None:
    """Called at the end of every epoch."""
    if self.shuffle:
      self._compute_batch_indices()

  def __len__(self) -> int:
    """Return the number of batches per epoch."""
    return int(np.ceil(self.n_samples / self.batch_size))

  def __getitem__(self, idx: int) -> Tuple[Dict[str, tf.Tensor], tf.Tensor]:
    """
    Generate one batch of data.

    Args:
      idx: Batch index

    Returns:
      Tuple of (input_dict, targets) where input_dict contains 'image_data' and 'numerical'
    """
    # Use pre-computed batch indices
    batch_indices = self._batch_indices[idx]

    # Direct numpy indexing (faster than tf.gather for small batches)
    batch_images = self.images[batch_indices]
    batch_numerical = self.numerical[batch_indices]
    batch_targets = self.targets[batch_indices]

    # Convert to tensors
    batch_images = tf.constant(batch_images, dtype=tf.float32)
    batch_numerical = tf.constant(batch_numerical, dtype=tf.float32)
    batch_targets = tf.constant(batch_targets, dtype=tf.float32)

    if self.augment:
      batch_images = self.augmentation(batch_images, training=True)

    return {'image_data': batch_images, 'numerical': batch_numerical}, batch_targets


class DataGenerator(BaseDataGenerator):
  pass


class ConcatenatedSequence(tf.keras.utils.Sequence):
  """Concatenate multiple Keras Sequences into a single Sequence."""

  def __init__(self, sequences: List[tf.keras.utils.Sequence]):
    self.sequences = sequences
    self._lengths = [len(seq) for seq in sequences]
    self._cumulative_lengths = np.cumsum(self._lengths)

  def __len__(self) -> int:
    return int(self._cumulative_lengths[-1])

  def __getitem__(self, idx: int):
    seq_idx = np.searchsorted(self._cumulative_lengths, idx, side='right')

    if seq_idx == 0:
      local_idx = idx
    else:
      local_idx = idx - self._cumulative_lengths[seq_idx - 1]

    return self.sequences[seq_idx][local_idx]

  def on_epoch_end(self) -> None:
    for seq in self.sequences:
      if hasattr(seq, 'on_epoch_end'):
        seq.on_epoch_end()