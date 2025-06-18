from typing import List, Dict, Optional, Any
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


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
    colors: Optional[Dict[str, str]] = None
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
      '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b',
      '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'
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
    self,
    ax: plt.Axes,
    metric_name: str,
    train_values: List[float],
    val_values: Optional[List[float]] = None
  ) -> None:
    """Plot a single metric with optional validation values.
    
    Args:
        ax: Matplotlib axes to plot on
        metric_name: Name of the metric to plot
        train_values: List of training metric values
        val_values: Optional list of validation metric values
    """
    epochs = range(len(train_values))
    ax.plot(epochs, train_values, label=f'Training {metric_name}',
            color=self.colors[metric_name], linewidth=2)
    
    if val_values and self.track_validation:
      ax.plot(epochs, val_values, label=f'Validation {metric_name}',
              color=self.colors[f'val_{metric_name}'], linewidth=2, linestyle='--')
    
    # Add early stopping indicator if applicable
    if self.early_stopping_epoch is not None:
      ax.axvline(x=self.early_stopping_epoch, color='red', linestyle=':',
                label='Early Stopping')
    
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
        print(f"  Final value: {values[-1]:.5f}")
        print(f"  Mean: {np.mean(values_array):.5f}")
        print(f"  Std: {np.std(values_array):.5f}")
        print(f"  Min: {np.min(values_array):.5f}")
        print(f"  Max: {np.max(values_array):.5f}")
  
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