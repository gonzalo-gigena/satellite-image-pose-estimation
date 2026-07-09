import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from config.model_config import ModelConfig
from utils.conversions import angular_error_deg, quaternion_to_rotation_matrix
from utils.model_helpers import generate_output_path


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


def plot_weight_distributions(config: ModelConfig, cnn_branch):
  """Plot weight distributions for all conv layers in a 2-column grid."""

  sequential = getattr(cnn_branch, 'cnn_layers', None) or getattr(cnn_branch, 'layers', None)

  conv_layers = [
      layer for layer in sequential.layers
      if isinstance(layer, tf.keras.layers.Conv2D)
  ]

  n_layers = len(conv_layers)
  n_cols = 2
  n_rows = int(np.ceil(n_layers / n_cols))

  fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 4 * n_rows))
  fig.suptitle('Weight Distributions per Layer', fontsize=14)

  axes = np.array(axes).flatten()

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

  for j in range(n_layers, len(axes)):
    axes[j].set_visible(False)

  plt.tight_layout()

  output_path = generate_output_path(config, 'layers', prefix='weight_distributions_')
  plt.savefig(output_path, dpi=150)
  plt.close()


def plot_filters(config: ModelConfig, cnn_branch, max_filters_per_layer=None, min_display_size=16):
  """Visualize filters from all conv layers in a 2-column grid.

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

    scale = max(1, min_display_size // max(f_h, f_w))
    d_h, d_w = f_h * scale, f_w * scale

    pad = 2

    grid_cols = int(np.ceil(np.sqrt(n_filters)))
    grid_rows = int(np.ceil(n_filters / grid_cols))

    grid_h = grid_rows * (d_h + pad) + pad
    grid_w = grid_cols * (d_w + pad) + pad

    grid = np.ones((grid_h, grid_w, 3))

    for i in range(n_filters):
      row = i // grid_cols
      col = i % grid_cols
      y = row * (d_h + pad) + pad
      x = col * (d_w + pad) + pad

      if is_rgb:
        filt = weights[:, :, :, i]
      else:
        channel = weights[:, :, 0, i]
        filt = np.stack([channel] * 3, axis=-1)

      if scale > 1:
        filt = np.repeat(np.repeat(filt, scale, axis=0), scale, axis=1)

      grid[y:y + d_h, x:x + d_w, :] = filt

    return grid

  n_layers = len(layer_data)
  n_cols = 2
  n_rows = int(np.ceil(n_layers / n_cols))

  fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 8 * n_rows))
  axes = np.array(axes).flatten()

  for i, data in enumerate(layer_data):
    grid = build_layer_grid(data)

    title = (
        f"{data['name']}"
    )

    axes[i].imshow(grid, interpolation='nearest')
    axes[i].set_title(title, fontsize=20, fontweight='bold')
    axes[i].axis('off')

  for j in range(n_layers, len(axes)):
    axes[j].set_visible(False)

  fig.suptitle('All Convolutional Filters', fontsize=14, fontweight='bold')
  plt.tight_layout()

  output_path = generate_output_path(config, 'layers', prefix='filters_')
  plt.savefig(output_path, dpi=150, bbox_inches='tight')
  plt.close()


def _draw_camera_frustum(
    ax: Axes3D,
    R: np.ndarray,
    origin: np.ndarray,
    scale: float = 0.4,
    color: str = '#333333',
    alpha: float = 0.15,
) -> None:
  """Draw a camera frustum projected along the rotation's Z axis.

  Draws the image-plane rectangle and four lines connecting it to the
  camera centre, mimicking a standard camera frustum diagram.

  Args:
    ax:     3D matplotlib axes.
    R:      Rotation matrix (3, 3) – Z column is the optical axis.
    origin: Camera centre in world space (3,).
    scale:  Distance from origin to image plane along optical axis.
    color:  Colour for edges and face.
    alpha:  Face transparency.
  """
  w, h = 0.32, 0.24
  corners_cam = np.array([
      [-w, -h, scale],
      [w, -h, scale],
      [w, h, scale],
      [-w, h, scale],
  ])                                                # (4, 3)

  corners_world = (R @ corners_cam.T).T + origin   # (4, 3)

  rect = Poly3DCollection(
      [corners_world],
      alpha=alpha,
      facecolor=color,
      edgecolor=color,
      linewidth=1.2,
  )
  ax.add_collection3d(rect)

  for corner in corners_world:
    ax.plot(
        [origin[0], corner[0]],
        [origin[1], corner[1]],
        [origin[2], corner[2]],
        color=color,
        linewidth=0.8,
        alpha=0.6,
    )


def _gather_samples(dataset, n: int) -> tuple:
  """Pull n random samples from a tf.data dataset using random skips.

  Args:
    dataset: tf.data.Dataset yielding (inputs_dict, labels) batches.
    n:       Number of samples to randomly select.

  Returns:
    Tuple of (inputs dict with numpy arrays, labels numpy array).
  """
  inputs_collected = {'image_data': [], 'numerical': []}
  labels_collected = []

  # Get dataset size
  total_batches = sum(1 for _ in dataset)
  selected_batches = np.random.choice(total_batches, size=min(n, total_batches), replace=False)

  for idx in selected_batches:
    batch_inputs, batch_labels = next(iter(dataset.skip(int(idx)).take(1)))

    # Pick a single random sample within the batch
    batch_size = batch_labels.shape[0]
    sample_idx = np.random.randint(0, batch_size)

    inputs_collected['image_data'].append(batch_inputs['image_data'].numpy()[sample_idx:sample_idx + 1])
    inputs_collected['numerical'].append(batch_inputs['numerical'].numpy()[sample_idx:sample_idx + 1])
    labels_collected.append(batch_labels.numpy()[sample_idx:sample_idx + 1])

  inputs_out = {
      'image_data': np.concatenate(inputs_collected['image_data'], axis=0)[:n],
      'numerical': np.concatenate(inputs_collected['numerical'], axis=0)[:n],
  }
  labels_out = np.concatenate(labels_collected, axis=0)[:n]
  return inputs_out, labels_out


def _save_frustum_plot(
    preds: np.ndarray,
    labels: np.ndarray,
    split_name: str,
    config,
    n_samples: int,
) -> None:
  """Save the 3D frustum plot for all samples in a 2-column grid.

  Samples are arranged left-to-right, top-to-bottom across 2 columns.
  A single column is used only when n_samples == 1.

  Args:
    preds:      Predicted quaternions (n, 4).
    labels:     Ground truth quaternions (n, 4).
    split_name: 'training' or 'validation'.
    config:     ModelConfig for output path.
    n_samples:  Number of samples to plot.
  """
  origin = np.zeros(3)
  n_cols = 1 if n_samples == 1 else 2
  n_rows = int(np.ceil(n_samples / n_cols))

  fig = plt.figure(figsize=(7 * n_cols, 4 * n_rows))
  fig.suptitle(
      f'{split_name.capitalize()} – Satellite Rotation Prediction',
      fontsize=14,
      fontweight='bold',
  )

  gs = gridspec.GridSpec(
      n_rows, n_cols,
      figure=fig,
      hspace=0.1,
      wspace=0.05,
  )

  for i in range(n_samples):
    row = i // n_cols
    col = i % n_cols

    q_true = labels[i]
    q_pred = preds[i]
    err = angular_error_deg(q_pred, q_true)

    R_true = quaternion_to_rotation_matrix(q_true)
    R_pred = quaternion_to_rotation_matrix(q_pred)

    err_color = (
        '#4CAF50' if err < 5 else
        '#FF9800' if err < 15 else
        '#F44336'
    )

    ax3d = fig.add_subplot(gs[row, col], projection='3d')

    _draw_camera_frustum(
        ax3d, R_true, origin,
        scale=0.4, color='#c62828', alpha=0.20,
    )
    _draw_camera_frustum(
        ax3d, R_pred, origin,
        scale=0.4, color='#0d47a1', alpha=0.20,
    )

    ax3d.scatter(*origin, color='black', s=60, zorder=10)

    ax3d.view_init(elev=20, azim=45)
    ax3d.set_title(
        f'Sample {i + 1}  |  Angular error: {err:.2f}°',
        fontsize=9,
        color=err_color,
        fontweight='bold',
        pad=4,
    )

    lim = 0.6
    ax3d.set_xlim(-lim, lim)
    ax3d.set_ylim(-lim, lim)
    ax3d.set_zlim(-lim, lim)
    ax3d.set_xlabel('X', fontsize=7)
    ax3d.set_ylabel('Y', fontsize=7)
    ax3d.set_zlabel('Z', fontsize=7)
    ax3d.tick_params(labelsize=6)

    legend_elements = [
        mpatches.Patch(color='#F44336', alpha=0.8, label='Ground truth'),
        mpatches.Patch(color='#2196F3', alpha=0.8, label='Predicted'),
    ]
    ax3d.legend(
        handles=legend_elements,
        loc='upper left',
        fontsize=7,
        framealpha=0.7,
    )

  fig.subplots_adjust(top=0.93, bottom=0.02, left=0.02, right=0.98)

  output_path = generate_output_path(config, split_name, 'frustum')
  fig.savefig(output_path.with_suffix('.png'), dpi=150, bbox_inches='tight')
  plt.close(fig)


def _save_frames_plot(
    images: np.ndarray,
    split_name: str,
    config,
    n_samples: int,
    frames: int,
) -> None:
  """Save the burst frame thumbnails for all samples.

  One row of F frames per sample with the sample index as row label
  and frame index as column header.

  Args:
    images:     Input burst images (n, F, H, W, C).
    split_name: 'training' or 'validation'.
    config:     ModelConfig for output path.
    n_samples:  Number of samples to plot.
    frames:     Number of frames per sample (F).
  """
  fig, axes = plt.subplots(
      n_samples, frames,
      figsize=(2.5 * frames, 2.5 * n_samples),
      squeeze=False,
  )
  fig.suptitle(
      f'{split_name.capitalize()} – Frames',
      fontsize=14,
      fontweight='bold',
      y=1.01,
  )

  for i in range(n_samples):
    for f in range(frames):
      ax = axes[i, f]
      frame_img = images[i, f].astype(np.float32)

      lo, hi = frame_img.min(), frame_img.max()
      if hi > lo:
        frame_img = (frame_img - lo) / (hi - lo)

      if frame_img.shape[-1] == 1:
        ax.imshow(frame_img[..., 0], cmap='gray', vmin=0, vmax=1)
      else:
        ax.imshow(np.clip(frame_img, 0, 1))

      ax.axis('off')

    axes[i, 0].set_ylabel(
        f'Sample {i + 1}',
        fontsize=8,
        fontweight='bold',
        color='#444444',
        labelpad=4,
    )

  fig.tight_layout()
  output_path = generate_output_path(config, split_name, 'frames_')
  fig.savefig(output_path.with_suffix('.png'), dpi=150, bbox_inches='tight')
  plt.close(fig)


def plot_prediction_examples(
    model,
    train_ds,
    val_ds,
    config: ModelConfig,
    n_samples: int = 4,
) -> None:
  """Visualise predicted vs ground-truth satellite rotation and burst frames.

  Saves two separate PNG files per split:

  1. ``predictions_{split}`` – 3D frustum subplots in a 2-column grid
     (1 column when n_samples == 1) showing ground truth (red) and
     predicted (blue) orientations with geodesic angular error in the title.

  2. ``frames_{split}`` – grid of burst frame thumbnails (n_samples rows,
     F columns) with sample index as row label and frame index as column
     header.

  Args:
    model:     Compiled Keras model returning unit quaternions (qw,qx,qy,qz).
    train_ds:  tf.data.Dataset for training   (batched).
    val_ds:    tf.data.Dataset for validation (batched).
    config:    ModelConfig – provides output path and number of frames.
    n_samples: Number of samples to draw from each split.
  """
  frames = config.frames
  splits = {
      'training': _gather_samples(train_ds, n_samples),
      'validation': _gather_samples(val_ds, n_samples),
  }

  for split_name, (inputs, labels) in splits.items():

    preds = model.predict(
        {'image_data': inputs['image_data'], 'numerical': inputs['numerical']},
        verbose=0,
    )                                                          # (n, 4)
    preds = preds / np.linalg.norm(preds, axis=-1, keepdims=True)

    images = inputs['image_data']                              # (n, F, H, W, C)

    _save_frustum_plot(preds, labels, split_name, config, n_samples)
    _save_frames_plot(images, split_name, config, n_samples, frames)
