import json
import time
from collections import defaultdict
from typing import List, Tuple

import tensorflow as tf
from tensorflow.keras.callbacks import Callback, History
from tensorflow.keras.losses import Loss
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Optimizer

from config.model_config import ModelConfig
from data.generator import ConcatenatedSequence, DataGenerator
from data.loader import DataLoader, TrainValTestData
from utils.model_helpers import (calculate_max_sequences, generate_filename,
                                 generate_output_path, get_data_generator,
                                 get_data_loader, get_loss_function,
                                 get_metrics, get_model, get_optimizer,
                                 plot_quaternion_loss)

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
    # Initialize model and data
    self._config = config
    self._model: Model = get_model(config)
    self._data_loader: DataLoader = get_data_loader(self._config)
    self._optimizer: Optimizer = get_optimizer(config.optimizer, config.lr)
    self._loss_function: Loss = get_loss_function(config.loss)
    self._metrics = get_metrics()
    self._is_compiled: bool = False

  def _create_generators(self, data: TrainValTestData) -> Tuple[DataGenerator, DataGenerator, DataGenerator]:
    """Create training and validation data generators."""
    train_generator = get_data_generator(
        data=data.train,
        batch_size=self._config.batch_size,
        shuffle=True,
    )

    val_generator = get_data_generator(
        data=data.val,
        batch_size=self._config.batch_size,
        shuffle=False
    )

    test_generator = get_data_generator(
        data=data.test,
        batch_size=self._config.batch_size,
        shuffle=False
    )

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

    callbacks.append(
        RotationMetricsCallback(
            metrics_to_track=[self._config.monitor_metric],
            track_validation=True,
            path=path,
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
    """Execute the training pipeline."""

    self._compile_model()
    callbacks = self._create_callbacks()

    files = self._data_loader.load_files()
    max_sequences = calculate_max_sequences(self._config)
    files_per_chunk = max_sequences * self._config.frames

    all_metrics = defaultdict(list)

    global_epoch = 0
    chunk_times = []
    test_generators = []

    total_start = time.perf_counter()
    for b in range(0, len(files), files_per_chunk):
      print(f'Chunk {b//files_per_chunk}')
      burst_chunk = files[b:b + files_per_chunk]
      data: TrainValTestData = self._data_loader.load_data(burst_chunk)

      # Create data generators
      train_gen, val_gen, test_gen = self._create_generators(data)
      test_generators.append(test_gen)

      history, chunk_time = self._train_chunk(
          train_gen, val_gen, callbacks, global_epoch
      )
      chunk_times.append(chunk_time)

      global_epoch += self._config.epochs

      # merge history for this chunk into a single run
      for k, v in history.history.items():
        all_metrics[k].extend(v)

    total_elapsed = time.perf_counter() - total_start
    all_metrics['chunk_time_sec'] = chunk_times
    all_metrics['total_time_sec'] = total_elapsed

    final_test_generator = ConcatenatedSequence(test_generators)
    test_results = self._model.evaluate(
        final_test_generator,
        verbose=2,
        return_dict=True,
    )

    # merge history for this chunk into a single run
    all_metrics['test'] = test_results

    self._save_metrics(all_metrics)
    plot_quaternion_loss(all_metrics, self._config)
