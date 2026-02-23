import json
import time
from typing import List, Tuple

import tensorflow as tf
from tensorflow.keras.callbacks import Callback, History
from tensorflow.keras.losses import Loss
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Optimizer

from config.model_config import ModelConfig
from data.generator import DataGenerator
from data.loader import DataLoader, TrainValTestData
from utils.model_helpers import (generate_filename, generate_output_path,
                                 get_loss_function, get_metrics, get_model,
                                 get_optimizer, plot_quaternion_loss)

from .callbacks import EnhancedModelCheckpoint


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
    # Initialize model and data
    self._config = config
    self._model: Model = get_model(config)
    self._data_loader: DataLoader = DataLoader(self._config)
    self._optimizer: Optimizer = get_optimizer(config.optimizer, config.lr)
    self._loss_function: Loss = get_loss_function(config.loss)
    self._metrics = get_metrics()
    self._is_compiled: bool = False

  def _create_generators(self, data: TrainValTestData) -> Tuple[DataGenerator, DataGenerator, DataGenerator]:
    """Create training and validation data generators."""
    train_generator = DataGenerator(
        data=data.train,
        config=self._config,
        shuffle=True,
    ).get_dataset()

    val_generator = DataGenerator(
        data=data.val,
        config=self._config,
        shuffle=False
    ).get_dataset()

    test_generator = DataGenerator(
        data=data.test,
        config=self._config,
        shuffle=False
    ).get_dataset()

    return train_generator, val_generator, test_generator

  def _create_callbacks(self) -> List[Callback]:
    """Create enhanced callbacks for training."""
    callbacks = []
    path = generate_filename(self._config)

    # Enhanced model checkpoint
    callbacks.append(
        EnhancedModelCheckpoint(
            log_dir=self._config.log_dir,
            max_models=self._config.max_models,
            monitor=self._config.monitor_metric,
            resume_training=self._config.resume_training,
            path=path
        )
    )

    # Early stopping
    callbacks.append(
        tf.keras.callbacks.EarlyStopping(
            monitor=self._config.monitor_metric,
            patience=15,
            restore_best_weights=True,
            mode=self._config.monitor_mode,
            verbose=1
        )
    )

    # Reduce learning rate on plateau
    callbacks.append(
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor=self._config.monitor_metric,
            mode=self._config.monitor_mode,
            factor=0.5,
            patience=10,
            min_lr=1e-7,
            verbose=1
        )
    )
    return callbacks

  def _compile_model(self) -> None:
    """Compile the model with optimizer, loss, and metrics."""
    if self._is_compiled:
      return

    self._model.compile(
        optimizer=self._optimizer,
        loss=self._loss_function,
        metrics=get_metrics(),
    )
    self._is_compiled = True

  def _save_metrics(self, all_metrics: dict) -> None:
    metrics_path = generate_output_path(self._config, 'metrics')
    with metrics_path.open('w', encoding='utf-8') as f:
      json.dump(all_metrics, f, indent=2)

  def _train_chunk(
      self,
      train: DataGenerator,
      val: DataGenerator,
      callbacks: List[Callback],
      global_epoch: int
  ) -> Tuple[History, float]:
    """Train model on a single data chunk.

    Args:
      generators: Data generators for the chunk.
      callbacks: Training callbacks.
      global_epoch: Starting epoch number.

    Returns:
      Tuple of training history and elapsed time.
    """
    start_time = time.perf_counter()

    history = self._model.fit(
        train,
        validation_data=val,
        epochs=global_epoch + self._config.epochs,
        initial_epoch=global_epoch,
        callbacks=callbacks,
        verbose=2,
    )

    elapsed = time.perf_counter() - start_time
    return history, elapsed

  def train(self) -> None:
    """Execute the training pipeline using tf.data streaming."""

    self._compile_model()

    total_start = time.perf_counter()

    # Load full dataset (only file paths + numerical data, no images in memory)
    files = self._data_loader.load_files()
    data: TrainValTestData = self._data_loader.load_data(files)

    # Build streaming datasets
    train_ds, val_ds, test_ds = self._create_generators(data)

    callbacks = self._create_callbacks()

    history = self._model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=self._config.epochs,
        callbacks=callbacks,
        verbose=2,
    )

    test_results = self._model.evaluate(
        test_ds,
        verbose=2,
        return_dict=True,
    )

    total_elapsed = time.perf_counter() - total_start

    all_metrics = dict(history.history)
    all_metrics['total_time_sec'] = total_elapsed
    all_metrics['test'] = test_results

    self._save_metrics(all_metrics)
    plot_quaternion_loss(all_metrics, self._config)
