import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Union

import tensorflow as tf


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
      resume_training: bool = False,
      path: str = '',
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
      resume_training: Whether to resume from best checkpoint
      **kwargs: Additional arguments for ModelCheckpoint
    """

    # Determine the base directory (two levels above current file)
    base_dir = Path(__file__).resolve().parents[2]
    self.log_dir = base_dir / log_dir
    self.log_dir.mkdir(parents=True, exist_ok=True)

    self.checkpoints_dir = self.log_dir / f'checkpoints/{path}'
    self.checkpoints_dir.mkdir(parents=True, exist_ok=True)
    self.file_format = '.weights.h5' if save_weights_only else '.keras'

    # Create dynamic filepath pattern
    epoch_part = '{epoch:03d}'
    metric_name = monitor
    metric_value = f'{{{monitor}:.6f}}'
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
    self.resume_training = resume_training
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

    if self.resume_training and self.checkpoints:
      best_checkpoint = self.checkpoints[0]
      try:
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
    # epoch is 0-indexed
    if (epoch + 1) % 5 != 0:
      return

    # Get current metric value
    current = logs.get(self.monitor)
    if current is None:
      return

    self._sort_checkpoints()

    should_save = False

    if len(self.checkpoints) < self.max_models:
      should_save = True
    else:
      worst_metric = self.checkpoints[self.max_models - 1].metric_value

      if self.mode == 'min':
        should_save = current < worst_metric
      else:
        should_save = current > worst_metric

    if should_save:
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
      self.checkpoints = self.checkpoints[:self.max_models]

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
      best_checkpoint = self.checkpoints[0]  # Fixed: define before use
      model.load_weights(best_path, by_name=True, skip_mismatch=True)  # Fixed: use model param
      if self.verbose > 0:
        print(
            f'\nLoaded best model from epoch {best_checkpoint.epoch} '
            f'with {self.metric_name}: {best_checkpoint.metric_value:.6f}'
        )
      return True
    except Exception as e:
      print(f'Failed to load model: {e}')
      return False
