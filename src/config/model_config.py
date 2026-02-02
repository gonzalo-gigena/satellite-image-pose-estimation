from dataclasses import dataclass
from typing import Optional


# Immutable
@dataclass(frozen=True)
class ModelConfig:
  """Configuration class for ML model training parameters."""

  # Data parameters
  data_path: str
  train_split: float = 0.8
  validation_split: float = 0.1
  test_split: float = 0.1
  branch_type: Optional[str] = None

  # Training parameters
  batch_size: int = 128
  frames: int = 3
  image_height: int = 102
  image_width: int = 102
  channels: int = 1
  epochs: int = 100
  lr: float = 0.001
  optimizer: str = 'adam'
  loss: str = 'quaternion'
  load_weights: bool = False
  train_weights: bool = False
  resume_training: bool = True

  # Output parameters
  log_dir: str = './logs'

  # Miscellaneous parameters
  seed: int = 42

  # Hardcoded Params
  max_models: int = 10
  monitor_metric: str = 'quaternion_loss'
  monitor_mode: str = 'min'
  use_lr_scheduler: bool = True
  memory_limit_gb: float = 62  # 64 is for mendieta, adjust as necessary

  def __post_init__(self) -> None:
    """Validate configuration after initialization."""
    # Set default branch_type for relative_pose model
    if self.branch_type is None:
      object.__setattr__(self, 'branch_type', 'cnnAspp')
    self._validate()

  def _validate(self) -> None:
    """Validate configuration parameters."""
    # Validate split ratios
    if not 0 <= self.train_split <= 1:
      raise ValueError(f'Train split must be between 0 and 1, got {self.train_split}.')

    if not 0 <= self.validation_split <= 1:
      raise ValueError(f'Validation split must be between 0 and 1, got {self.validation_split}.')

    if self.train_split + self.validation_split > 1:
      raise ValueError(
          f'Train split ({self.train_split}) + validation split ({self.validation_split}) '
          f'must not exceed 1.0, got {self.train_split + self.validation_split}.'
      )

    # Validate optimizer
    valid_optimizers = ['adam', 'sgd', 'rmsprop']
    if self.optimizer not in valid_optimizers:
      raise ValueError(f'Invalid optimizer: {self.optimizer}. Must be one of {valid_optimizers}.')

    # Validate loss function
    valid_losses = ['quaternion', 'angular', 'detailed', 'geodesic']
    if self.loss not in valid_losses:
      raise ValueError(f'Invalid loss function: {self.loss}. Must be one of {valid_losses}.')

    # Validate relative_pose specific parameters
    valid_branch_types = ['cnnA', 'cnnAspp', 'cnnB', 'cnnBspp']
    if self.branch_type not in valid_branch_types:
      raise ValueError(
          f'Invalid branch type: {self.branch_type}. '
          f'Must be one of {valid_branch_types}.'
      )
    if self.load_weights and self.channels != 3:
      raise ValueError(
          f'When loading weights, channels must be 3, got {self.channels}.'
      )

    # Validate training flags
    if self.train_weights and not self.load_weights:
      raise ValueError('When using --train_weights (-tw), --load_weights (-lw) must also be set.')

    # Validate positive numeric parameters
    if self.batch_size <= 0:
      raise ValueError(f'Batch size must be positive, got {self.batch_size}.')

    if self.epochs <= 0:
      raise ValueError(f'Epochs must be positive, got {self.epochs}.')

    if self.lr <= 0:
      raise ValueError(f'Learning rate must be positive, got {self.lr}.')

    if self.frames <= 0:
      raise ValueError(f'Frames must be positive, got {self.frames}.')

    if self.channels <= 0:
      raise ValueError(f'Channels must be positive, got {self.channels}.')

    if self.image_height <= 0 or self.image_width <= 0:
      raise ValueError(
          f'Image dimensions must be positive, got height={self.image_height}, width={self.image_width}.'
      )
