from typing import List
import tensorflow as tf
import matplotlib.pyplot as plt

PLT_PATH = 'model_training_metrics_plot.png'

# TODO: Add configuration
def get_callbacks() -> List[tf.keras.callbacks.Callback]:
  """Return default training callbacks.

  Returns:
    List of Keras callbacks for training monitoring and optimization
  """

  return [
    RotationMetricsCallback(),
    tf.keras.callbacks.EarlyStopping(
      monitor='quaternion_loss',
      patience=10,
      restore_best_weights=True,
      mode='min'
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
      monitor='quaternion_loss',
      factor=0.5,
      patience=5,
      min_lr=1e-6
    )
  ]

class RotationMetricsCallback(tf.keras.callbacks.Callback):
  def __init__(self):
    self.num_epochs = 0
    
    # Initialize lists for all metrics
    self._loss = []
    self._mae = []
    self._quaternion_loss = []
    self._angular_distance_loss = []
    self._detailed_distance_loss = []
    self._geodesic_loss = []
      
  def _plot_model_performance(self):
    plt.rcParams['figure.figsize'] = (36, 12)
    plt.rcParams['axes.spines.top'] = False
    plt.rcParams['axes.spines.right'] = False
    
    fig, axes = plt.subplots(2, 3, figsize=(24, 12))
    fig.suptitle('Model Training Performance', size=20)
    axes = axes.flatten()
    
    # Plot overall loss
    axes[0].plot(range(self.num_epochs), self._loss, label='Training loss', color='blue')
    axes[0].set_xlabel('Epoch', size=14)
    axes[0].set_ylabel('Loss', size=14)
    axes[0].set_title('Overall Loss')
    axes[0].legend()
    
    # Plot MAE
    axes[1].plot(range(self.num_epochs), self._mae, label='MAE', color='green')
    axes[1].set_xlabel('Epoch', size=14)
    axes[1].set_ylabel('MAE', size=14)
    axes[1].set_title('Mean Absolute Error')
    axes[1].legend()
    
    # Plot Quaternion Loss
    axes[2].plot(range(self.num_epochs), self._quaternion_loss, label='Quaternion Loss', color='red')
    axes[2].set_xlabel('Epoch', size=14)
    axes[2].set_ylabel('Loss', size=14)
    axes[2].set_title('Quaternion Loss')
    axes[2].legend()
    
    # Plot Angular Distance Loss
    axes[3].plot(range(self.num_epochs), self._angular_distance_loss, label='Angular Distance Loss', color='purple')
    axes[3].set_xlabel('Epoch', size=14)
    axes[3].set_ylabel('Loss', size=14)
    axes[3].set_title('Angular Distance Loss')
    axes[3].legend()
    
    # Plot Detailed Distance Loss
    axes[4].plot(range(self.num_epochs), self._detailed_distance_loss, label='Detailed Distance Loss', color='orange')
    axes[4].set_xlabel('Epoch', size=14)
    axes[4].set_ylabel('Loss', size=14)
    axes[4].set_title('Detailed Distance Loss')
    axes[4].legend()
    
    # Plot Geodesic Loss
    axes[5].plot(range(self.num_epochs), self._geodesic_loss, label='Geodesic Loss', color='brown')
    axes[5].set_xlabel('Epoch', size=14)
    axes[5].set_ylabel('Loss', size=14)
    axes[5].set_title('Geodesic Loss')
    axes[5].legend()
    
    plt.tight_layout()
    plt.savefig(PLT_PATH)
      
  def on_train_end(self, logs=None):
    
    # Print final metrics
    print('\nFinal Training Metrics:')
    print(f"Overall loss:            {self._loss[-1]:.5f}")
    print(f"MAE:                     {self._mae[-1]:.5f}")
    print(f"Quaternion loss:         {self._quaternion_loss[-1]:.5f}")
    print(f"Angular distance loss:   {self._angular_distance_loss[-1]:.5f}")
    print(f"Detailed distance loss:  {self._detailed_distance_loss[-1]:.5f}")
    print(f"Geodesic loss:           {self._geodesic_loss[-1]:.5f}")
    
    self._plot_model_performance()
      
  def on_epoch_end(self, epoch, logs=None):
    self.num_epochs += 1
    
    # Extract metrics from logs
    loss = logs.get('loss', 0)
    mae = logs.get('mae', 0)
    quaternion_loss = logs.get('quaternion_loss', 0)
    angular_distance_loss = logs.get('angular_distance_loss', 0)
    detailed_distance_loss = logs.get('detailed_distance_loss', 0)
    geodesic_loss = logs.get('geodesic_loss', 0)
    
    # Append to history
    self._loss.append(loss)
    self._mae.append(mae)
    self._quaternion_loss.append(quaternion_loss)
    self._angular_distance_loss.append(angular_distance_loss)
    self._detailed_distance_loss.append(detailed_distance_loss)
    self._geodesic_loss.append(geodesic_loss)