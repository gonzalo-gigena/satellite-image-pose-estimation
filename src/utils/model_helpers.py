import time
from pathlib import Path
from typing import List, Literal

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras.losses import Loss
from tensorflow.keras.optimizers import Optimizer

from config.model_config import ModelConfig
from losses.custom import (angular_distance_loss, detailed_distance_loss,
                           geodesic_loss, quaternion_loss)


def get_metrics() -> List[tf.keras.metrics.Metric]:
  """Return default metrics.

  Returns:
    List of Keras metrics for training monitoring and optimization
  """
  # return ['mae', quaternion_loss, angular_distance_loss, detailed_distance_loss, geodesic_loss]
  return [quaternion_loss]


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


def plot_weight_distributions(config: ModelConfig, cnn_branch):
  """Plot weight distributions for all conv layers."""

  sequential = getattr(cnn_branch, 'cnn_layers', None) or getattr(cnn_branch, 'layers', None)

  conv_layers = [
      layer for layer in sequential.layers
      if isinstance(layer, tf.keras.layers.Conv2D)
  ]

  fig, axes = plt.subplots(1, len(conv_layers), figsize=(20, 4))
  fig.suptitle('Weight Distributions per Layer', fontsize=14)

  for i, layer in enumerate(conv_layers):
    try:
      weights = layer.get_weights()[0].flatten()
      biases = layer.get_weights()[1].flatten()

      axes[i].hist(weights, bins=50, alpha=0.7, label='weights', color='blue')
      axes[i].hist(biases, bins=50, alpha=0.7, label='biases', color='red')
      axes[i].set_title(f'{layer.name}\nμ={weights.mean():.4f} σ={weights.std():.4f}')
      axes[i].legend(fontsize=7)
      axes[i].set_xlabel('Value')
    except Exception as e:
      axes[i].set_title(f'{layer.name}\n(Error: {e})')

  plt.tight_layout()

  output_path = generate_output_path(config, 'layers', prefix='weight_distributions_')
  plt.savefig(output_path, dpi=150)
  plt.close()


def visualize_all_filters(config: ModelConfig, cnn_branch, max_filters_per_layer=64, min_display_size=16):
  """Visualize filters from all conv layers in a single figure.

  Args:
    cnn_branch: The CNN branch model
    max_filters_per_layer: Cap on filters shown per layer
    min_display_size: Minimum pixel size for each filter in the display
  """
  sequential = getattr(cnn_branch, 'cnn_layers', None) or getattr(cnn_branch, 'layers', None)

  conv_layers = [
      layer for layer in sequential.layers
      if isinstance(layer, tf.keras.layers.Conv2D)
  ]

  if not conv_layers:
    raise ValueError('No Conv2D layers found')

  layer_data = []
  for layer in conv_layers:
    weights = layer.get_weights()[0]

    w_min, w_max = weights.min(), weights.max()
    weights_norm = (weights - w_min) / (w_max - w_min + 1e-8)

    n_filters = weights.shape[-1]
    if max_filters_per_layer is not None:
      n_filters = min(n_filters, max_filters_per_layer)

    layer_data.append({
        'name': layer.name,
        'weights': weights_norm,
        'shape': weights.shape,
        'n_filters': n_filters,
    })

  def build_layer_grid(data):
    weights = data['weights']
    n_filters = data['n_filters']
    f_h, f_w = weights.shape[0], weights.shape[1]
    is_rgb = weights.shape[2] == 3

    # Scale up small filters so they're visible
    scale = max(1, min_display_size // max(f_h, f_w))
    d_h, d_w = f_h * scale, f_w * scale

    pad = 2

    n_cols = int(np.ceil(np.sqrt(n_filters)))
    n_rows = int(np.ceil(n_filters / n_cols))

    grid_h = n_rows * (d_h + pad) + pad
    grid_w = n_cols * (d_w + pad) + pad

    grid = np.ones((grid_h, grid_w, 3))

    for i in range(n_filters):
      row = i // n_cols
      col = i % n_cols
      y = row * (d_h + pad) + pad
      x = col * (d_w + pad) + pad

      if is_rgb:
        filt = weights[:, :, :, i]
      else:
        channel = weights[:, :, 0, i]
        filt = np.stack([channel] * 3, axis=-1)

      # Scale up the filter using nearest-neighbor
      if scale > 1:
        filt = np.repeat(np.repeat(filt, scale, axis=0), scale, axis=1)

      grid[y:y + d_h, x:x + d_w, :] = filt

    return grid

  grids = [build_layer_grid(data) for data in layer_data]

  # --- Stack vertically, no excess whitespace ---
  max_width = max(g.shape[1] for g in grids)
  label_height = 40
  gap = 15

  total_height = sum(g.shape[0] + label_height + gap for g in grids)

  canvas = np.ones((total_height, max_width, 3))

  y_offset = 0
  label_positions = []

  for grid, data in zip(grids, layer_data):
    label_positions.append({
        'y': y_offset + label_height // 2,
        'text': (
            f"{data['name']} — "
            f"{data['n_filters']}/{data['shape'][-1]} filters, "
            f"size {data['shape'][0]}×{data['shape'][1]}, "
            f"C_in={data['shape'][2]}"
        ),
    })

    y_start = y_offset + label_height
    x_start = (max_width - grid.shape[1]) // 2
    canvas[
        y_start:y_start + grid.shape[0],
        x_start:x_start + grid.shape[1],
        :
    ] = grid

    y_offset = y_start + grid.shape[0] + gap

  # Crop unused bottom space
  canvas = canvas[:y_offset]

  dpi = 150
  fig_w = max(max_width / dpi * 3, 6)
  fig_h = max(y_offset / dpi * 3, 4)
  fig_w = min(fig_w, 25)
  fig_h = min(fig_h, 50)

  fig, ax = plt.subplots(1, 1, figsize=(fig_w, fig_h))
  ax.imshow(canvas, interpolation='nearest')

  for lp in label_positions:
    ax.text(
        5, lp['y'], lp['text'],
        fontsize=9, fontweight='bold',
        color='black',
        verticalalignment='center',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.9)
    )

  ax.axis('off')
  fig.suptitle('All Convolutional Filters', fontsize=14, fontweight='bold')
  plt.tight_layout()

  output_path = generate_output_path(config, 'layers', prefix='filters_')
  plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
  plt.close()
