import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf


class RotationMetricsCallback(tf.keras.callbacks.Callback):
  """Callback for tracking and visualizing metrics during training.

  This callback tracks specified metrics and creates visualizations of the training progress.
  It supports both training and validation metrics, and provides statistical analysis.

  Attributes:
      metrics_to_track: List of metric names to track
      plot_path: Path where the metrics plot will be saved
      figsize: Tuple of (width, height) for the plot figure
      colors: Dictionary mapping metric names to their plot colors
      track_validation: Whether to track validation metrics
      early_stopping_epoch: Epoch at which early stopping occurred (if any)
  """

  def __init__(
      self,
      metrics_to_track: List[str],
      plot_path: str = 'model_training_metrics_plot.png',
      figsize: tuple = (22, 12),
      track_validation: bool = False,
      colors: Optional[Dict[str, str]] = None,
  ):
    """Initialize the callback.

    Args:
      metrics_to_track: List of metric names to track (e.g., ['loss', 'mae', 'quaternion_loss'])
      plot_path: Path where to save the metrics plot
      figsize: Figure size as (width, height) tuple
      track_validation: Whether to track validation metrics
      colors: Dictionary mapping metric names to plot colors
    """
    super().__init__()
    self.metrics_to_track = metrics_to_track
    self.plot_path = Path(plot_path)
    self.figsize = figsize
    self.track_validation = track_validation
    self.early_stopping_epoch: Optional[int] = None

    # Default colors for metrics (using a color cycle)
    default_colors = [
        '#1f77b4',
        '#ff7f0e',
        '#2ca02c',
        '#d62728',
        '#9467bd',
        '#8c564b',
        '#e377c2',
        '#7f7f7f',
        '#bcbd22',
        '#17becf',
    ]

    # Create color mapping for metrics
    self.colors = colors or {}
    for i, metric in enumerate(self.metrics_to_track):
      if metric not in self.colors:
        self.colors[metric] = default_colors[i % len(default_colors)]
      if self.track_validation and f'val_{metric}' not in self.colors:
        self.colors[f'val_{metric}'] = default_colors[i % len(default_colors)]

    # Initialize metric tracking
    self._initialize_metrics()

  def _initialize_metrics(self) -> None:
    """Initialize lists for tracking metrics."""
    self.num_epochs = 0
    self.metrics: Dict[str, List[float]] = {metric: [] for metric in self.metrics_to_track}

    if self.track_validation:
      self.metrics.update({f'val_{metric}': [] for metric in self.metrics_to_track})

  def _plot_metric(
      self, ax: plt.Axes, metric_name: str, train_values: List[float], val_values: Optional[List[float]] = None
  ) -> None:
    """Plot a single metric with optional validation values.

    Args:
        ax: Matplotlib axes to plot on
        metric_name: Name of the metric to plot
        train_values: List of training metric values
        val_values: Optional list of validation metric values
    """
    epochs = range(len(train_values))
    ax.plot(epochs, train_values, label=f'Training {metric_name}', color=self.colors[metric_name], linewidth=2)

    if val_values and self.track_validation:
      ax.plot(
          epochs,
          val_values,
          label=f'Validation {metric_name}',
          color=self.colors[f'val_{metric_name}'],
          linewidth=2,
          linestyle='--',
      )

    # Add early stopping indicator if applicable
    if self.early_stopping_epoch is not None:
      ax.axvline(x=self.early_stopping_epoch, color='red', linestyle=':', label='Early Stopping')

    ax.set_xlabel('Epoch', size=12)
    ax.set_ylabel(metric_name.replace('_', ' ').title(), size=12)
    ax.set_title(metric_name.replace('_', ' ').title(), size=14)
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.3)

  def _plot_model_performance(self) -> None:
    """Create and save the performance visualization plot."""
    plt.style.use('default')
    plt.rcParams['figure.figsize'] = self.figsize
    plt.rcParams['axes.spines.top'] = False
    plt.rcParams['axes.spines.right'] = False

    n_metrics = len(self.metrics_to_track)
    n_cols = min(3, n_metrics)
    n_rows = (n_metrics + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=self.figsize)
    fig.suptitle('Model Training Performance', size=20, y=1.02)
    axes = axes.flatten()

    # Plot each metric
    for i, metric_name in enumerate(self.metrics_to_track):
      val_values = self.metrics.get(f'val_{metric_name}') if self.track_validation else None
      self._plot_metric(axes[i], metric_name, self.metrics[metric_name], val_values)

    # Remove empty subplots
    for j in range(i + 1, len(axes)):
      fig.delaxes(axes[j])

    plt.tight_layout()
    self.plot_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(self.plot_path, bbox_inches='tight', dpi=300)
    plt.close()

  def _print_metric_statistics(self) -> None:
    """Print statistical analysis of the metrics."""
    print('\nMetric Statistics:')
    print('-' * 50)

    for metric_name in self.metrics_to_track:
      values = self.metrics[metric_name]
      if values:  # Only print if we have values
        values_array = np.array(values)
        print(f"\n{metric_name.replace('_', ' ').title()}:")
        print(f'  Final value: {values[-1]:.5f}')
        print(f'  Mean: {np.mean(values_array):.5f}')
        print(f'  Std: {np.std(values_array):.5f}')
        print(f'  Min: {np.min(values_array):.5f}')
        print(f'  Max: {np.max(values_array):.5f}')

  def on_train_end(self, logs: Optional[Dict[str, float]] = None) -> None:
    """Called at the end of training.

    Args:
        logs: Dictionary of metrics for the last epoch
    """
    self._print_metric_statistics()
    self._plot_model_performance()

  def on_epoch_end(self, epoch: int, logs: Optional[Dict[str, float]] = None) -> None:
    """Called at the end of each epoch.

    Args:
        epoch: Current epoch number
        logs: Dictionary of metrics for the current epoch
    """
    if logs is None:
      return

    self.num_epochs += 1

    # Track metrics
    for metric_name in self.metrics.keys():
      value = logs.get(metric_name, 0.0)
      self.metrics[metric_name].append(value)

    # Check for early stopping
    if logs.get('stop_training', False):
      self.early_stopping_epoch = epoch


@dataclass
class Checkpoint:
  """Represents a model checkpoint with metadata."""

  filepath: str
  epoch: int
  metric_value: float
  metric_name: str

  def to_dict(self) -> Dict:
    """Convert to dictionary for JSON serialization."""
    return asdict(self)

  @classmethod
  def from_dict(cls, data: Dict) -> 'Checkpoint':
    """Create from dictionary."""
    return cls(**data)


class EnhancedModelCheckpoint(tf.keras.callbacks.ModelCheckpoint):
  """Enhanced model checkpoint that extends ModelCheckpoint with checkpoint management."""

  def __init__(
      self,
      log_dir: str,
      max_models: int = 10,
      monitor: str = 'quaternion_loss',
      mode: str = 'min',
      save_best_only: bool = True,
      save_freq: Union[int, str] = 'epoch',
      save_weights_only: bool = False,
      verbose: int = 1,
      load_best_on_start: bool = False,
      channels: int = 1,
      frames: int = 3,
      image_height: int = 102,
      image_width: int = 102,
      **kwargs,
  ):
    """Initialize the enhanced checkpoint callback.

    Args:
      log_dir: Directory to store model checkpoints
      max_models: Maximum number of models to keep
      monitor: Metric to monitor
      mode: 'min' or 'max'
      save_best_only: Whether to save only when monitored metric improves
      save_weights_only: Whether to save only weights
      save_freq: Save frequency in epochs
      verbose: Verbosity level
      load_best_on_start: Whether to load best model when training starts
      **kwargs: Additional arguments for ModelCheckpoint
    """
    self.frames = frames
    self.image_height = image_height
    self.image_width = image_width
    self.channels = channels
    self.log_dir = Path(log_dir)
    self.checkpoints_dir = self.log_dir / 'checkpoints'
    self.checkpoints_dir.mkdir(parents=True, exist_ok=True)
    self.file_format = '.weights.h5' if save_weights_only else '.keras'

    # Create dynamic filepath pattern
    epoch_part = '{epoch:03d}'
    metric_name = monitor  # e.g., 'quaternion_loss'
    metric_value = f'{{{monitor}:.6f}}'  # e.g., '{quaternion_loss:.6f}'
    filename = f'{epoch_part}_{metric_name}_{metric_value}{self.file_format}'
    filepath = str(self.checkpoints_dir / filename)

    # Initialize parent ModelCheckpoint
    super().__init__(
        filepath=filepath,
        monitor=monitor,
        mode=mode,
        save_best_only=save_best_only,
        save_weights_only=save_weights_only,
        save_freq=save_freq,
        verbose=verbose,
        **kwargs,
    )

    self.mode = mode
    self.max_models = max_models
    self.metric_name = monitor
    self.load_best_on_start = load_best_on_start
    self.checkpoints: List[Checkpoint] = []
    self._load_existing_checkpoints()

    # Update best value if we have existing checkpoints
    if self.checkpoints and save_best_only:
      self.best = self.checkpoints[0].metric_value
      if self.verbose > 0:
        print(f'Found existing best model with {self.metric_name}: {self.best:.6f}')

  def on_train_begin(self, logs=None):
    """Called at the beginning of training."""
    super().on_train_begin(logs)

    if self.load_best_on_start and self.checkpoints:
      best_checkpoint = self.checkpoints[0]
      try:
        self.model.build({})
        # Load only weights to avoid optimizer state mismatch
        self.model.load_weights(best_checkpoint.filepath, skip_mismatch=True)
        if self.verbose > 0:
          print(
              f'\nLoaded best model from epoch {best_checkpoint.epoch} '
              f'with {self.metric_name}: {best_checkpoint.metric_value:.6f}'
          )
          print(f'Continuing training from this checkpoint...\n')
      except Exception as e:
        print(f'Warning: Failed to load best model: {e}')
        print('Starting training from current model state...')

  def _load_existing_checkpoints(self) -> None:
    """Load existing checkpoints from directory."""
    if self.checkpoints_dir.exists():
      for file in os.listdir(self.checkpoints_dir):
        if file.endswith(self.file_format):
          try:
            # Parse filename: epoch_metricname_value.weights.h5
            parts = file.replace(self.file_format, '').split('_')
            epoch = int(parts[0])
            metric_value = float(parts[-1])
            metric_name = '_'.join(parts[1:-1])

            checkpoint = Checkpoint(
                filepath=str(self.checkpoints_dir / file),
                epoch=epoch,
                metric_value=metric_value,
                metric_name=metric_name,
            )
            self.checkpoints.append(checkpoint)
          except (ValueError, IndexError):
            # Skip files that don't match expected pattern
            continue

      # Sort by metric value
      self._sort_checkpoints()

      if self.verbose > 0 and self.checkpoints:
        print(f'\nFound {len(self.checkpoints)} existing checkpoint(s)')

  def _sort_checkpoints(self) -> None:
    """Sort checkpoints by metric value."""
    self.checkpoints.sort(key=lambda x: x.metric_value, reverse=(self.mode == 'max'))

  def _save_model(self, epoch, batch, logs):
    """Override parent's _save_model to add checkpoint management."""
    # Get current metric value
    current = logs.get(self.monitor)
    if current is None:
      return

    # Call parent's save method
    super()._save_model(epoch, batch, logs)

    # Add checkpoint to our list
    checkpoint_path = self.filepath.format(epoch=epoch + 1, monitor=self.monitor, **logs)

    checkpoint = Checkpoint(
        filepath=checkpoint_path, epoch=epoch + 1, metric_value=current, metric_name=self.metric_name
    )
    self.checkpoints.append(checkpoint)

    self._sort_checkpoints()

    # Manage checkpoint count
    self._cleanup_old_checkpoints()

  def _cleanup_old_checkpoints(self) -> None:
    """Remove old checkpoints to maintain max_models limit."""
    if len(self.checkpoints) > self.max_models:
      # Remove worst models
      models_to_remove = self.checkpoints[self.max_models:]
      for cp in models_to_remove:
        try:
          if os.path.exists(cp.filepath):
            os.remove(cp.filepath)
            if self.verbose > 0:
              print(f'\nRemoved old checkpoint: {cp.filepath}')
        except OSError as e:
          print(f'Failed to remove {cp.filepath}: {e}')

      # Keep only the best models
      self.checkpoints = self.checkpoints[: self.max_models]

  def get_best_model_path(self) -> Optional[str]:
    """Get the filepath of the best model."""
    if not self.checkpoints:
      return None
    return self.checkpoints[0].filepath

  def load_best_model(self, model: tf.keras.Model) -> bool:
    """Load the best model weights into the given model.

    Args:
      model: The model to load weights into

    Returns:
      True if model was loaded successfully, False otherwise
    """
    best_path = self.get_best_model_path()
    if best_path is None:
      return False

    try:
      self.model.load_weights(best_checkpoint.filepath, by_name=True, skip_mismatch=True)
      if self.verbose > 0:
        best_checkpoint = self.checkpoints[0]
        print(
            f'\nLoaded best model from epoch {best_checkpoint.epoch} '
            f'with {self.metric_name}: {best_checkpoint.metric_value:.6f}'
        )
      return True
    except Exception as e:
      print(f'Failed to load model: {e}')
      return False
