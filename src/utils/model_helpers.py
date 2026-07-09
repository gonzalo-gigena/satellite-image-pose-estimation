import time
from pathlib import Path
from typing import List, Literal

import tensorflow as tf
from tensorflow.keras.losses import Loss
from tensorflow.keras.optimizers import Optimizer

from config.model_config import ModelConfig
from losses.custom import (angular_distance_loss, angular_error_degrees,
                           detailed_distance_loss, geodesic_loss,
                           quaternion_loss)


def get_metrics() -> List[tf.keras.metrics.Metric]:
  """Return default metrics.

  Returns:
    List of Keras metrics for training monitoring and optimization
  """
  return [quaternion_loss, angular_error_degrees]


def generate_filename(config: ModelConfig) -> str:
  path = (
      f'{config.image_height}_'
      f'{config.image_width}_'
      f'{config.frames}_'
      f'{int(config.load_weights)}_'
      f'{int(config.train_weights)}_'
      f'{config.channels}_'
      f'{config.degrees}_'
      f'{config.branch_type}'
  )

  return path


def generate_output_path(config: ModelConfig, folder: Literal['plots', 'metrics', 'layers'], prefix='') -> Path:
  base_dir = Path(__file__).resolve().parents[2]
  dir = base_dir / config.log_dir / folder

  extension = 'json' if folder == 'metrics' else 'png'

  run_name = prefix + generate_filename(config) if len(prefix) else generate_filename(config)
  output_path = dir / f'{run_name}_{time.time()}.{extension}'
  output_path.parent.mkdir(parents=True, exist_ok=True)
  return output_path


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
