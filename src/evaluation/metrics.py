import os
import json
import numpy as np
import matplotlib.pyplot as plt

def save_history_metrics(history_metrics, save_dir='./src/evaluation'):
  """Save history metrics to a JSON file."""
  os.makedirs(save_dir, exist_ok=True)
  metrics_file = os.path.join(save_dir, f'history_metrics.json')
  
  # Convert numpy types to Python native types for JSON serialization
  def convert_to_native_types(obj):
    if isinstance(obj, np.integer):
      return int(obj)
    elif isinstance(obj, np.floating):
      return float(obj)
    elif isinstance(obj, np.ndarray):
      return obj.tolist()
    return obj
  
  # Process the metrics dictionary
  processed_metrics = {}
  for key, value in history_metrics.items():
    if isinstance(value, dict):
      processed_metrics[key] = {
          k: convert_to_native_types(v) for k, v in value.items()
      }
    else:
      processed_metrics[key] = convert_to_native_types(value)
  
  with open(metrics_file, 'w') as f:
    json.dump(processed_metrics, f, indent=2)
  
  return metrics_file

def extract_history_metrics(history):
  """
  Extract and process common metrics from training history.
  
  Args:
    history: Keras history object containing training metrics
      
  Returns:
    Dictionary containing processed metrics from training history
  """
  history_metrics = {}
  
  # List of common metric prefixes and their display names
  metric_mapping = {
    'loss': 'Loss',
    'accuracy': 'Accuracy',
    'acc': 'Accuracy',  # some models use 'acc' instead of 'accuracy'
    'precision': 'Precision',
    'recall': 'Recall',
    'auc': 'AUC',
    'mae': 'Mean Absolute Error',
    'mse': 'Mean Squared Error',
    'rmse': 'Root Mean Squared Error',
    'val_loss': 'Validation Loss',
    'val_accuracy': 'Validation Accuracy',
    'val_acc': 'Validation Accuracy',
    'val_precision': 'Validation Precision',
    'val_recall': 'Validation Recall',
    'val_auc': 'Validation AUC',
    'val_mae': 'Validation MAE',
    'val_mse': 'Validation MSE',
    'val_rmse': 'Validation RMSE'
  }
  
  try:
      # Convert history.history to dict if it's not already
      history_dict = history.history if isinstance(history, object) else history
      
      # Process each metric in history
      for metric_name, values in history_dict.items():
        # Store the raw values
        history_metrics[metric_name] = {
          'values': values,
          'display_name': metric_mapping.get(metric_name, metric_name.replace('_', ' ').title()),
          'final_value': float(values[-1]),
          'best_value': float(min(values) if 'loss' in metric_name.lower() 
                            else max(values)),
          'best_epoch': int(np.argmin(values) if 'loss' in metric_name.lower() 
                          else np.argmax(values))
        }
        
        # Calculate additional statistics
        history_metrics[metric_name].update({
          'mean': float(np.mean(values)),
          'std': float(np.std(values)),
          'min': float(np.min(values)),
          'max': float(np.max(values)),
          'improvement': float(values[-1] - values[0]),
          'improvement_percentage': float((values[-1] - values[0]) / values[0] * 100)
        })

      # Add training summary
      history_metrics['training_summary'] = {
        'total_epochs': len(history_dict[list(history_dict.keys())[0]]),
        'metrics_tracked': list(history_dict.keys()),
        'has_validation': any('val_' in metric for metric in history_dict.keys())
      }
      
      # Add convergence analysis
      for metric_name in history_dict.keys():
        if 'loss' in metric_name.lower():
          values = history_dict[metric_name]
          # Check if training has converged (you can adjust the threshold)
          convergence_threshold = 1e-4
          converged = False
          convergence_epoch = None
          
          for i in range(len(values)-5):  # Check last 5 epochs
            if abs(values[i+4] - values[i]) < convergence_threshold:
              converged = True
              convergence_epoch = i
              break
          
          history_metrics[metric_name]['convergence'] = {
            'converged': converged,
            'convergence_epoch': convergence_epoch,
            'final_delta': float(abs(values[-1] - values[-2]))
          }

  except Exception as e:
    print(f"Error extracting history metrics: {str(e)}")
    history_metrics['error'] = str(e)
  
  return history_metrics

def plot_history_metrics(history_metrics, save_dir='./src/evaluation'):
  """
  Create detailed plots for all metrics in training history.
  
  Args:
      history_metrics: Dictionary containing processed history metrics
      save_dir: Directory to save the plots
  """
  os.makedirs(save_dir, exist_ok=True)

  # Group related metrics for plotting
  metric_groups = {}
  
  for metric_name in history_metrics.keys():
    if metric_name not in ['training_summary', 'error']:
      # Group metrics by their base name (without 'val_' prefix)
      base_name = metric_name.replace('val_', '')
      if base_name not in metric_groups:
        metric_groups[base_name] = []
      metric_groups[base_name].append(metric_name)
  
  # Create plots for each metric group
  for base_name, metrics in metric_groups.items():
    plt.figure(figsize=(12, 6))
    
    for metric_name in metrics:
      values = history_metrics[metric_name]['values']
      display_name = history_metrics[metric_name]['display_name']
      plt.plot(values, label=display_name)
      
      # Add markers for best value
      best_epoch = history_metrics[metric_name]['best_epoch']
      best_value = history_metrics[metric_name]['best_value']
      plt.plot(best_epoch, best_value, 'o', label=f'Best {display_name}')
    
    plt.title(f'{history_metrics[metrics[0]]["display_name"]} Over Time')
    plt.xlabel('Epoch')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    
    # Save the plot
    plt.savefig(os.path.join(save_dir, f'{base_name}_history.png'))
    plt.close()
  
  # Create a summary plot of all final metric values
  plt.figure(figsize=(12, 6))
  final_metrics = {
    metric_name: data['final_value'] 
    for metric_name, data in history_metrics.items() 
    if metric_name not in ['training_summary', 'error']
  }
  
  plt.bar(final_metrics.keys(), final_metrics.values())
  plt.title('Final Metric Values')
  plt.xticks(rotation=45)
  plt.tight_layout()
  plt.savefig(os.path.join(save_dir, f'final_metrics_summary.png'))
  plt.close()

# Additional function to plot quaternion components distribution
def plot_quaternion_distribution(predictions, targets, save_dir='./evaluation'):
  fig, axes = plt.subplots(2, 2, figsize=(15, 10))
  axes = axes.ravel()
  
  # Split predictions and targets into individual quaternions
  pred_q0, pred_q1 = np.split(predictions, 2, axis=-1)
  true_q0, true_q1 = np.split(targets, 2, axis=-1)
  
  # Plot distributions for each quaternion component
  for i in range(4):
    axes[i].hist(pred_q0[:, i], bins=30, alpha=0.5, label='Predicted Q0', color='blue')
    axes[i].hist(true_q0[:, i], bins=30, alpha=0.5, label='True Q0', color='red')
    axes[i].set_title(f'Component {i+1} Distribution')
    axes[i].legend()
    axes[i].grid(True)
  
  plt.tight_layout()
  plt.savefig(os.path.join(save_dir, f'final_metrics.png'))
  plt.close()

def analyze_history(history):
  metrics = extract_history_metrics(history)
  
  save_history_metrics(metrics)
  
  plot_history_metrics(metrics)