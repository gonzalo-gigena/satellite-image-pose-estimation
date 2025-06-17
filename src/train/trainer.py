from typing import Tuple, Dict, Any
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import History
from tensorflow.keras.optimizers import Optimizer
from tensorflow.keras.losses import Loss

from utils.model_utils import get_data_loader, get_model, get_train_generator
from utils.optimizers import get_optimizer
from utils.loss_functions import get_loss_function
from .callbacks import get_default_callbacks
from config.model_config import ModelConfig


class ModelTrainer:
  """Handles model training and evaluation pipeline."""
  
  def __init__(self, config: ModelConfig) -> None:
    """Initialize trainer with configuration."""
    self.config = config
    self.model: Model = get_model(config.model)
    self.data: Dict[str, Dict[str, Any]] = self._load_data()
    self.optimizer: Optimizer = get_optimizer(config.optimizer, config.lr)
    self.loss_function: Loss = get_loss_function(config.loss)
    
  def _load_data(self) -> Dict[str, Dict[str, Any]]:
    """Load and prepare data for training.
    
    Returns:
      Dictionary containing train/val data splits
    """
    data_loader = get_data_loader(self.config)
    return data_loader.load_data()
    
  def _create_generators(self) -> Tuple[Any, Any]:  # Replace Any with actual generator types
    """Create training and validation data generators.
    
    Returns:
      Tuple of (train_generator, val_generator)
    """
    train_generator = get_train_generator(
      self.data['train'],
      self.config.batch_size,
      self.config.model,
      shuffle=False,
    )
    
    val_generator = get_train_generator(
      self.data['val'],
      self.config.batch_size,
      self.config.model,
      shuffle=False
    )
    
    return train_generator, val_generator
    
  def train(self) -> Tuple[Model, History]:
    """Execute the training pipeline.
    
    Returns:
      Tuple of (trained_model, training_history)
    """
    # Create data generators
    train_generator, val_generator = self._create_generators()
    
    # Compile model
    self.model.compile(
      optimizer=self.optimizer,
      loss=self.loss_function,
      metrics=['mae']
    )
    
    # Train model
    history = self.model.fit(
      train_generator,
      validation_data=val_generator,
      epochs=self.config.epochs,  # Fixed: was self.args.epochs
      callbacks=get_default_callbacks()
    )
    
    return self.model, history
    
  def save_model(self) -> None:
    """Save the trained model if path is specified."""
    if self.config.model_save_path:
      self.model.save(self.config.model_save_path)  # Fixed: was self.args.model_save_path
      print(f"Model saved to {self.config.model_save_path}")