import numpy as np
import tensorflow as tf

from config.model_config import ModelConfig
from data.loader import DataSplit


class DataGenerator:
  """
  tf.data based streaming dataset.
  """

  def __init__(
      self,
      data: DataSplit,
      config: ModelConfig,
      shuffle: bool = False,
      augment: bool = False,
      debug: bool = False
  ) -> None:

    self.filepaths = np.array(data.images, dtype=object)[data.indices]
    self.numerical = data.numerical[data.indices]
    self.targets = data.targets[data.indices]

    self.image_height = config.image_height
    self.image_width = config.image_width
    self.channels = config.channels
    self.frames = config.frames
    self.batch_size = config.batch_size
    self.shuffle = shuffle
    self.augment = augment

    self.debug = debug
    self._debug_counter = 0

    self.dataset = self._build_dataset()

  def _load_burst(self, burst_paths, numerical, target):

    def _load_image(path):
      image = tf.io.read_file(path)
      image = tf.image.decode_jpeg(image, channels=self.channels)
      image = tf.image.resize(image, [self.image_height, self.image_width])
      image = tf.cast(image, tf.float32) / 255.0
      return image

    images = tf.map_fn(
        _load_image,
        burst_paths,
        fn_output_signature=tf.float32
    )

    if self.augment:
      images = tf.image.random_brightness(images, 0.2)
      images = tf.image.random_contrast(images, 0.8, 1.2)

    if self.debug:
      # Check correct frame count
      tf.debugging.assert_equal(
          tf.shape(images)[0],
          self.frames,
          message='Incorrect number of frames in burst'
      )

      # Check pixel range
      tf.debugging.assert_greater_equal(
          tf.reduce_min(images),
          0.0,
          message='Pixel values below 0'
      )

      tf.debugging.assert_less_equal(
          tf.reduce_max(images),
          1.0,
          message='Pixel values above 1'
      )

      # Check non-empty image
      tf.debugging.assert_greater(
          tf.reduce_mean(images),
          0.01,
          message='Image appears empty or black'
      )

    return {'image_data': images, 'numerical': numerical}, target

  def _build_dataset(self):

    ds = tf.data.Dataset.from_tensor_slices(
        (self.filepaths, self.numerical, self.targets)
    )

    if self.shuffle:
      ds = ds.shuffle(buffer_size=len(self.filepaths))

    ds = ds.map(
        self._load_burst,
        num_parallel_calls=tf.data.AUTOTUNE
    )

    ds = ds.batch(self.batch_size)
    ds = ds.prefetch(tf.data.AUTOTUNE)

    return ds

  def get_dataset(self) -> tf.data.Dataset:
    return self.dataset
