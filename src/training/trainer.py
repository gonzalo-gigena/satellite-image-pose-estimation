from typing import Tuple

import tensorflow as tf
from tensorflow.keras.losses import Loss
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Optimizer

from config.model_config import ModelConfig
from data.generator import DataGenerator
from data.loader import DataLoader, TrainValData
from utils.model_helpers import (calculate_max_sequences, generate_path,
                                 get_data_loader, get_loss_function,
                                 get_metrics, get_model, get_optimizer,
                                 get_train_generator)

from .callbacks import EnhancedModelCheckpoint, RotationMetricsCallback


class ModelTrainer:
  """Enhanced trainer with advanced model management and training features."""

  def __init__(
      self,
      config: ModelConfig,
  ) -> None:
    """Initialize the enhanced trainer.

    Args:
        config: Model configuration
        max_models: Maximum number of models to keep
        monitor_metric: Metric to monitor for model selection
        monitor_mode: 'min' or 'max' for metric optimization
        use_lr_scheduler: Whether to use learning rate scheduler
    """
    self.config = config

    # Initialize model and data
    self.model: Model = get_model(config)
    self.data_loader: DataLoader = get_data_loader(self.config)
    self.optimizer: Optimizer = get_optimizer(config.optimizer, config.lr)
    self.loss_function: Loss = get_loss_function(config.loss)

  def _create_generators(self, data: TrainValData) -> Tuple[DataGenerator, DataGenerator]:
    """Create training and validation data generators."""
    train_generator = get_train_generator(
        data['train'],
        self.config.batch_size,
        self.config.model,
        shuffle=True,
    )

    val_generator = get_train_generator(data['val'], self.config.batch_size, self.config.model, shuffle=False)

    return train_generator, val_generator

  def _create_callbacks(self) -> list:
    """Create enhanced callbacks for training."""
    callbacks = []
    path = generate_path(self.config)

    # Enhanced model checkpoint
    callbacks.append(
        EnhancedModelCheckpoint(
            log_dir=self.config.log_dir,
            max_models=self.config.max_models,
            monitor=self.config.monitor_metric,
            resume_training=self.config.resume_training,
            path=path
        )
    )

    callbacks.append(
        RotationMetricsCallback(
            metrics_to_track=[self.config.monitor_metric],
            track_validation=True,
            path=path,
        )
    )

    # Early stopping
    callbacks.append(
        tf.keras.callbacks.EarlyStopping(
            monitor=self.config.monitor_metric,
            patience=15,
            restore_best_weights=True,
            mode=self.config.monitor_mode,
            verbose=1
        )
    )

    # Reduce learning rate on plateau
    callbacks.append(
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor=self.config.monitor_metric,
            mode=self.config.monitor_mode,
            factor=0.5,
            patience=5,
            min_lr=1e-7,
            verbose=1
        )
    )
    return callbacks

  def train(self) -> None:
    """Execute the training pipeline."""
    # Get metrics and callbacks
    custom_metrics = get_metrics()
    callbacks = self._create_callbacks()

    # Compile model
    self.model.compile(optimizer=self.optimizer, loss=self.loss_function, metrics=custom_metrics)

    files = self.data_loader.load_files()
    max_sequences = calculate_max_sequences(self.config)
    files_per_chunk = max_sequences * self.config.frames
    for i in range(0, len(files), files_per_chunk):
      chunk = files[i:i + files_per_chunk]

      data: TrainValData = self.data_loader.load_data(chunk)

      # Create data generators
      train_generator, val_generator = self._create_generators(data)
      # Train model
      history = self.model.fit(
          train_generator,
          validation_data=val_generator,
          epochs=self.config.epochs,
          callbacks=callbacks,
          verbose=1
      )
