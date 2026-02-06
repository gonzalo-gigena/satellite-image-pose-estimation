import time
from pathlib import Path
from typing import List, Literal

import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.losses import Loss
from tensorflow.keras.optimizers import Optimizer

from config.model_config import ModelConfig
from data.generator import DataGenerator
from data.loader import DataLoader, DataSplit
from losses.custom import (angular_distance_loss, detailed_distance_loss,
                           geodesic_loss, quaternion_loss)
from models.relative_pose import RelativePoseModel


def get_metrics() -> List[tf.keras.metrics.Metric]:
  """Return default metrics.

  Returns:
    List of Keras metrics for training monitoring and optimization
  """
  # return ['mae', quaternion_loss, angular_distance_loss, detailed_distance_loss, geodesic_loss]
  return ['mae', quaternion_loss]


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


def plot_quaternion_loss(
    metrics: dict,
    config: ModelConfig
):
  train_loss = metrics['quaternion_loss']
  val_loss = metrics['val_quaternion_loss']

  epochs = range(1, len(train_loss) + 1)

  plt.figure(figsize=(10, 6))

  plt.plot(epochs, train_loss, label='Train Quaternion Loss')
  plt.plot(
      epochs,
      val_loss,
      linestyle='--',
      label='Validation Quaternion Loss'
  )

  plt.xlabel('Epoch')
  plt.ylabel('Loss')

  # Line legend (only explains curves)
  plt.legend(loc='upper right')

  # Metadata text box (bottom-left, bold labels)
  info_text = (
      r'$\mathbf{Resolution:}$ '
      f'{config.image_height}×{config.image_width}\n'
      r'$\mathbf{Frames:}$ '
      f'{config.frames}\n'
      r'$\mathbf{Channels:}$ '
      f'{config.channels}\n'
      r'$\mathbf{Degrees:}$ '
      f'{config.degrees}\n'
      r'$\mathbf{Branch:}$ '
      f'{config.branch_type}\n'
      r'$\mathbf{Load\ Weights:}$ '
      f'{config.load_weights}\n'
      r'$\mathbf{Train\ Weights:}$ '
      f'{config.train_weights}'
  )

  plt.gca().text(
      0.02,
      0.02,
      info_text,
      transform=plt.gca().transAxes,
      fontsize=9,
      verticalalignment='bottom',
      horizontalalignment='left',
      bbox=dict(
          boxstyle='round,pad=0.4',
          facecolor='white',
          edgecolor='gray',
          alpha=0.85
      )
  )

  plt.grid(True)
  plt.tight_layout()

  path = generate_output_path(config, 'plots')
  plt.savefig(path, dpi=300)
  plt.close()


def generate_output_path(config: ModelConfig, prefix: Literal['plots', 'metrics']) -> Path:
  base_dir = Path(__file__).resolve().parents[2]
  metrics_dir = base_dir / prefix

  extension = 'json' if prefix == 'metrics' else 'png'

  run_name = generate_filename(config)
  metrics_path = metrics_dir / f'{run_name}_{time.time()}.{extension}'
  metrics_path.parent.mkdir(parents=True, exist_ok=True)
  return metrics_path


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


def get_model(config: ModelConfig) -> RelativePoseModel:
  """Select and return the appropriate model."""
  return RelativePoseModel(
      config.image_height,
      config.image_height,
      config.channels,
      config.frames,
      config.branch_type,
      config.load_weights,
      config.train_weights
  )


def get_data_loader(config: ModelConfig) -> DataLoader:
  """Select and return the appropriate data loader based on the matching method.

  Args:
    config: ModelConfig object containing all configuration parameters

  Returns:
    DataLoader: The appropriate data loader instance
  """
  return DataLoader(config)


def get_data_generator(
    data: DataSplit, batch_size: int, shuffle: bool = True, augment: bool = False
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
      data=data,
      shuffle=shuffle,
      batch_size=batch_size,
      augment=augment,
  )


def calculate_max_sequences(config: ModelConfig) -> int:
  """Calculate maximum sequences that fit within memory limit."""
  factor = 3
  dtype_bytes = 4  # float32

  memory_limit_bytes = (config.memory_limit_gb // factor) * (1024 ** 3)
  bytes_per_sequence = config.frames * config.image_height * config.image_width * config.channels * dtype_bytes
  max_sequences = int(memory_limit_bytes // bytes_per_sequence)

  return max_sequences
