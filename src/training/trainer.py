from typing import Tuple
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import History
from tensorflow.keras.optimizers import Optimizer
from tensorflow.keras.losses import Loss

from utils.model_helpers import (
  get_data_loader,
  get_model,
  get_train_generator, 
  get_loss_function,
  get_optimizer,
  get_metrics
)

from config.model_config import ModelConfig
from data.loader import TrainValData
from models.grayscale import GrayscaleDataGenerator

from .callbacks import (
  EnhancedModelCheckpoint,
  RotationMetricsCallback
)

class ModelTrainer:
  """Enhanced trainer with advanced model management and training features."""
  
  def __init__(
    self, 
    config: ModelConfig,
    max_models: int = 10,
    monitor_metric: str = 'quaternion_loss',
    monitor_mode: str = 'min',
    resume_training: bool = True,
    use_lr_scheduler: bool = True
  ) -> None:
    """Initialize the enhanced trainer.
    
    Args:
        config: Model configuration
        max_models: Maximum number of models to keep
        monitor_metric: Metric to monitor for model selection
        monitor_mode: 'min' or 'max' for metric optimization
        resume_training: Whether to resume from best checkpoint
        use_lr_scheduler: Whether to use learning rate scheduler
    """
    self.config = config
    self.max_models = max_models
    self.monitor_metric = monitor_metric
    self.monitor_mode = monitor_mode
    self.resume_training = resume_training
    self.use_lr_scheduler = use_lr_scheduler
    
    # Initialize model and data
    self.model: Model = get_model(config.model, config.channels, config.image_height, config.image_width)
    self.data: TrainValData = self._load_data()
    self.optimizer: Optimizer = get_optimizer(config.optimizer, config.lr)
    self.loss_function: Loss = get_loss_function(config.loss)
    
  def _load_data(self) -> TrainValData:
    """Load and prepare data for training."""
    data_loader = get_data_loader(self.config)
    return data_loader.load_data()
  
  def _create_generators(self) -> Tuple[GrayscaleDataGenerator, GrayscaleDataGenerator]:
    """Create training and validation data generators."""
    train_generator = get_train_generator(
        self.data['train'],
        self.config.batch_size,
        self.config.model,
        shuffle=True,
    )
    
    val_generator = get_train_generator(
        self.data['val'],
        self.config.batch_size,
        self.config.model,
        shuffle=False
    )
    
    return train_generator, val_generator
  
  def _create_callbacks(self) -> list:
    """Create enhanced callbacks for training."""
    callbacks = []
    
    # Enhanced model checkpoint
    callbacks.append(EnhancedModelCheckpoint(
      log_dir=self.config.log_dir,
      max_models=self.max_models,
      monitor=self.monitor_metric,
      load_best_on_start=True,
      channels=self.config.channels,
      frames=self.config.frames,
      image_height=self.config.image_height,
      image_width=self.config.image_width
    ))
    
    callbacks.append(RotationMetricsCallback(
      metrics_to_track=['loss', 'quaternion_loss'],
      track_validation=True
    ))
    
    # Early stopping
    callbacks.append(tf.keras.callbacks.EarlyStopping(
      monitor=self.monitor_metric,
      patience=15,
      restore_best_weights=True,
      mode=self.monitor_mode,
      verbose=1
    ))
    
    # Reduce learning rate on plateau
    callbacks.append(tf.keras.callbacks.ReduceLROnPlateau(
      monitor=self.monitor_metric,
      factor=0.5,
      patience=8,
      min_lr=1e-7,
      verbose=1
    ))
    return callbacks
  
  def train(self) -> Tuple[Model, History]:
    """Execute the enhanced training pipeline.
    
    Returns:
        Tuple of (trained_model, training_history)
    """
    # Create data generators
    train_generator, val_generator = self._create_generators()
    
    # Get metrics and callbacks
    custom_metrics = get_metrics()
    callbacks = self._create_callbacks()
    
    # Compile model
    self.model.compile(
        optimizer=self.optimizer,
        loss=self.loss_function,
        metrics=custom_metrics
    )

    # Train model
    history = self.model.fit(
        train_generator,
        validation_data=val_generator,
        epochs=self.config.epochs,
        callbacks=callbacks,
        verbose=1
    )
    
    return self.model, history