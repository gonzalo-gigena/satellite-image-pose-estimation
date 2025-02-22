from utils.model_utils import get_data_loader, get_model, get_train_generator
from utils.optimizers import get_optimizer
from utils.loss_functions import get_loss_function
from .callbacks import get_default_callbacks

class ModelTrainer:
  """Handles model training and evaluation pipeline."""
  
  def __init__(self, args):
    """Initialize trainer with command line arguments."""
    self.args = args
    self.model = get_model(args.model)
    self.data = self._load_data()
    self.optimizer = get_optimizer(args.optimizer, args.lr)
    self.loss_function = get_loss_function(args.loss)
    
  def _load_data(self):
    """Load and prepare data for training."""
    data_loader = get_data_loader(
      data_path=self.args.data_path,
      train_split=self.args.train_split,
      validation_split=self.args.validation_split,
      seed=self.args.seed,
      model=self.args.model,
      num_matches=self.args.num_matches
    )
    
    return data_loader.load_data()
    
  def _create_generators(self):
    """Create training and validation data generators."""
    train_generator = get_train_generator(
      self.data['train'],
      self.args.batch_size,
      self.args.model,
      shuffle=True
    )
    
    val_generator = get_train_generator(
      self.data['val'],
      self.args.batch_size,
      self.args.model,
      shuffle=False
    )
    
    return train_generator, val_generator
    
  def train(self):
    """Execute the training pipeline."""
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
      epochs=self.args.epochs,
      #callbacks=get_default_callbacks()
    )
    
    return self.model, history
    
  def save_model(self):
    """Save the trained model if path is specified."""
    if self.args.model_save_path:
      self.model.save(self.args.model_save_path)
      print(f"Model saved to {self.args.model_save_path}")