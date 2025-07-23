from dataclasses import dataclass


# Immutable
@dataclass(frozen=True)
class ModelConfig:
  """Configuration class for ML model training parameters."""

  # Data parameters
  data_path: str
  train_split: float = 0.8
  validation_split: float = 0.0
  model: str = 'grayscale'
  branch_type: str = None

  # Training parameters
  batch_size: int = 32
  frames: int = 3
  image_height: int = 102
  image_width: int = 102
  channels: int = 1
  epochs: int = 100
  lr: float = 0.001
  optimizer: str = 'adam'
  loss: str = 'quaternion'
  load_weights: bool = False

  # Output parameters
  log_dir: str = './logs'

  # Miscellaneous parameters
  seed: int = 42

  def __post_init__(self) -> None:
    """Validate configuration after initialization."""
    if self.model == 'relative_pose' and self.branch_type is None:
      object.__setattr__(self, 'branch_type', 'cnnAspp')
    self._validate()

  def _validate(self) -> None:
    """Validate configuration parameters."""
    if not 0 <= self.train_split <= 1:
      raise ValueError('Train split must be between 0 and 1.')

    if not 0 <= self.validation_split <= 1:
      raise ValueError('Validation split must be between 0 and 1.')

    if self.model not in ['grayscale', 'relative_pose']:
      raise ValueError(f'Invalid model: {self.model}')

    if self.optimizer not in ['adam', 'sgd', 'rmsprop']:
      raise ValueError(f'Invalid optimizer: {self.optimizer}')

    if self.loss not in ['quaternion', 'angular', 'detailed', 'geodesic']:
      raise ValueError(f'Invalid loss function: {self.loss}')

    if self.model == 'relative_pose':
      if self.branch_type not in ['cnnA', 'cnnAspp', 'cnnB', 'cnnBspp']:
        raise ValueError(f'Invalid branch type: {self.branch_type}')
      if self.load_weights and self.channels != 3:
        raise ValueError(f'When loading weights channels needs to be 3 not {self.channels}')
