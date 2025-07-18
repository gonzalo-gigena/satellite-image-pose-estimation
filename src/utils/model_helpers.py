from typing import List, Union

import tensorflow as tf
from tensorflow.keras.losses import Loss
from tensorflow.keras.optimizers import Optimizer

from config.model_config import ModelConfig
from data.generator import DataGenerator
from data.loader import DataLoader, DataSplit
from losses.custom import (angular_distance_loss, detailed_distance_loss,
                           geodesic_loss, quaternion_loss)
from models.grayscale import GrayscaleModel
from models.relative_pose import RelativePoseModel


def get_metrics() -> List[tf.keras.metrics.Metric]:
  """Return default metrics.

  Returns:
    List of Keras metrics for training monitoring and optimization
  """
  return ['mae', quaternion_loss, angular_distance_loss, detailed_distance_loss, geodesic_loss]


def get_loss_function(loss_name: str) -> Loss:
  """Select and return the loss function based on the given name"""
  loss_functions = {
      'quaternion': quaternion_loss,
      'angular': angular_distance_loss,
      'detailed': detailed_distance_loss,
      'geodesic': geodesic_loss,
  }

  loss_function = loss_functions.get(loss_name.lower())
  if loss_function is None:
    raise ValueError(f'Unsupported loss function: {loss_function}')

  return loss_function


def get_optimizer(optimizer_name: str, learning_rate: float) -> Optimizer:
  """Select and return the optimizer based on the given name."""
  optimizers = {
      'adam': tf.keras.optimizers.Adam,
      'sgd': tf.keras.optimizers.SGD,
      'rmsprop': tf.keras.optimizers.RMSprop,
  }

  optimizer_class = optimizers.get(optimizer_name.lower())
  if optimizer_class is None:
    raise ValueError(f'Unsupported optimizer: {optimizer_name}')

  return optimizer_class(learning_rate=learning_rate)


def get_model(config: ModelConfig) -> Union[GrayscaleModel, RelativePoseModel]:
  """Select and return the appropriate model."""
  if config.model == 'relative_pose':
    return RelativePoseModel(
        config.image_height,
        config.image_height,
        config.channels,
        config.frames,
        config.branch_type)
  return GrayscaleModel(config.image_height, config.image_width, config.channels, config.frames)


def get_data_loader(config: ModelConfig) -> DataLoader:
  """Select and return the appropriate data loader based on the matching method.

  Args:
    config: ModelConfig object containing all configuration parameters

  Returns:
    DataLoader: The appropriate data loader instance
  """
  return DataLoader(config)


def get_train_generator(
    data: DataSplit, batch_size: int, model: str, shuffle: bool = True, augment: bool = False
) -> DataGenerator:
  """Create and return the appropriate data generator based on the matching method.

  Args:
    data: Dictionary containing the training data with keys 'image_data', 'numerical', 'targets'
    batch_size: Batch size for training
    model: Model type identifier (currently unused but kept for future extensibility)
    shuffle: Whether to shuffle the data
    augment: Whether to apply data augmentation

  Returns:
    GrayscaleDataGenerator: The data generator instance
  """
  return DataGenerator(
      images=data['image_data'],
      numerical=data['numerical'],
      targets=data['targets'],
      shuffle=shuffle,
      batch_size=batch_size,
      augment=augment,
  )
