from dataclasses import dataclass
from typing import Optional

# Immutable
@dataclass(frozen=True)
class ModelConfig:
  """Configuration class for ML model training parameters."""
  # Data parameters
  data_path: str
  train_split: float = 0.8
  validation_split: float = 0.0
  model: str = 'grayscale'

  # Training parameters
  batch_size: int = 32
  epochs: int = 100
  lr: float = 0.001
  optimizer: str = 'adam'
  loss: str = 'quaternion'

  # Output parameters
  model_save_path: Optional[str] = None
  log_dir: str = './logs'

  # Miscellaneous parameters
  seed: int = 42

  def __post_init__(self) -> None:
    """Validate configuration after initialization."""
    self._validate()

  def _validate(self) -> None:
    """Validate configuration parameters."""
    if not 0 <= self.train_split <= 1:
      raise ValueError('Train split must be between 0 and 1.')

    if not 0 <= self.validation_split <= 1:
      raise ValueError('Validation split must be between 0 and 1.')

    if self.model not in ['grayscale', 'timeless']:
      raise ValueError(f'Invalid model: {self.model}')

    if self.optimizer not in ['adam', 'sgd', 'rmsprop']:
      raise ValueError(f'Invalid optimizer: {self.optimizer}')

    if self.loss not in ['quaternion', 'angular', 'detailed', 'geodesic']:
      raise ValueError(f'Invalid loss function: {self.loss}')