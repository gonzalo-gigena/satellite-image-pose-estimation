import os
from typing import List, Dict, Optional
from pathlib import Path
import tensorflow as tf
from dataclasses import dataclass, asdict

@dataclass
class Checkpoint:
  """Represents a model checkpoint with metadata."""
  filepath: str
  epoch: int
  metric_value: float
  metric_name: str
  
  def to_dict(self) -> Dict:
    """Convert to dictionary for JSON serialization."""
    return asdict(self)
  
  @classmethod
  def from_dict(cls, data: Dict) -> 'Checkpoint':
    """Create from dictionary."""
    return cls(**data)

class CheckpointManager:
  """Manages model checkpoints and keeps track of the best models."""
  
  def __init__(
    self, 
    log_dir: str, 
    max_models: int = 10,
    metric_name: str = 'quaternion_loss',
    mode: str = 'min'
  ):
    """Initialize the model manager.
    
    Args:
        log_dir: Directory to store model checkpoints
        max_models: Maximum number of models to keep
        metric_name: Metric to monitor for ranking models
        mode: 'min' for lower is better, 'max' for higher is better
    """
    self.log_dir = Path(log_dir)
    self.checkpoints_dir = self.log_dir / 'checkpoints'
    self.metadata_file = self.log_dir / 'model_metadata.json'
    self.max_models = max_models
    self.metric_name = metric_name
    self.mode = mode
    self.is_better = lambda x, y: x < y if mode == 'min' else x > y
    
    # Create directories
    self.checkpoints_dir.mkdir(parents=True, exist_ok=True)
    
    # Load existing metadata
    self.checkpoints: List[Checkpoint] = self._load_checkpoints()
      
  def _load_checkpoints(self) -> List[Checkpoint]:
    """Load existing model metadata by scanning checkpoint files in directory."""
    checkpoints = []
    # Scan directory and parse filenames
    if self.checkpoints_dir.exists():
      for file in os.listdir(self.checkpoints_dir):
        # Extract data from filename
        parts = file.split('_')
        epoch = int(parts[0])
        metric_name = parts[1]
        metric_value = float(parts[2].replace('.keras', ''))
        
        checkpoint = Checkpoint(
          filepath=str(self.checkpoints_dir / file),
          epoch=epoch,
          metric_value=metric_value,
          metric_name=metric_name,
        )
        checkpoints.append(checkpoint)
      
      # Sort by metric value (best first)
      checkpoints.sort(
        key=lambda x: x.metric_value,
        reverse=(self.mode == 'max')
      )
    
    return checkpoints
  
  def add_checkpoint(
    self, 
    model: tf.keras.Model, 
    epoch: int, 
    metric_value: float,
  ) -> bool:
    """Add a new model checkpoint.
    
    Args:
        model: The model to save
        epoch: Current epoch number
        metric_value: Value of the monitored metric
        timestamp: Timestamp string
        
    Returns:
        True if this is a new best model, False otherwise
    """
    # Create checkpoint filename
    checkpoint_name = f"{epoch:03d}_{self.metric_name}_{metric_value:.6f}.keras"
    checkpoint_path = self.checkpoints_dir / checkpoint_name
    
    # Save the model
    model.save(str(checkpoint_path))
    
    # Create checkpoint metadata
    checkpoint = Checkpoint(
      filepath=str(checkpoint_path),
      epoch=epoch,
      metric_value=metric_value,
      metric_name=self.metric_name
    )
    
    # Check if this is a new best model
    is_new_best = self._is_new_best(metric_value)
    
    # Add to checkpoints list
    self.checkpoints.append(checkpoint)
    
    # Sort by metric value (best first)
    self.checkpoints.sort(
      key=lambda x: x.metric_value,
      reverse=(self.mode == 'max')
    )
    
    # Keep only the best max_models
    if len(self.checkpoints) > self.max_models:
      # Remove worst models
      models_to_remove = self.checkpoints[self.max_models:]
      for cp in models_to_remove:
        try:
          os.remove(cp.filepath)
        except OSError as e:
          print(f"Failed to remove {cp.filepath}: {e}")
      
      # Keep only the best models
      self.checkpoints = self.checkpoints[:self.max_models]
    
    return is_new_best
  
  def _is_new_best(self, metric_value: float) -> bool:
    """Check if the given metric value is a new best."""
    if not self.checkpoints:
      return True
    
    best_value = self.checkpoints[0].metric_value
    return self.is_better(metric_value, best_value)
  
  def get_best_model_path(self) -> Optional[str]:
    """Get the filepath of the best model."""
    if not self.checkpoints:
      return None
    return self.checkpoints[0].filepath
  
  def load_best_model(self, model: tf.keras.Model) -> bool:
    """Load the best model weights into the given model.
    
    Args:
        model: The model to load weights into
        
    Returns:
        True if model was loaded successfully, False otherwise
    """
    best_path = self.get_best_model_path()
    if best_path is None:
      return False
    
    try:
      # Load the best model
      best_model = tf.keras.models.load_model(best_path)
      
      # Copy weights to the current model
      model.set_weights(best_model.get_weights())
      return True
        
    except Exception as e:
      return False
  
  def get_checkpoint_info(self) -> Dict:
    """Get information about all checkpoints."""
    if not self.checkpoints:
        return {"message": "No checkpoints found"}
    
    info = {
      "total_checkpoints": len(self.checkpoints),
      "best_model": {
        "epoch": self.checkpoints[0].epoch,
        "metric_value": self.checkpoints[0].metric_value,
        "filepath": self.checkpoints[0].filepath
      },
      "all_checkpoints": [
        {
          "epoch": cp.epoch,
          "metric_value": cp.metric_value,
          "filepath": cp.filepath,
        }
        for cp in self.checkpoints
      ]
    }
    return info