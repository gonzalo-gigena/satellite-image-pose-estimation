import argparse
from model_config import ModelConfig

def parse_args() -> ModelConfig:
  """Parse command line arguments and return ModelConfig object."""
  parser = argparse.ArgumentParser(
    description='Machine Learning Model Parameters',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter
  )

  # Data parameters
  _add_data_arguments(parser)
  
  # Training parameters
  _add_training_arguments(parser)
  
  # Output parameters
  _add_output_arguments(parser)
  
  # Miscellaneous parameters
  _add_misc_arguments(parser)

  args = parser.parse_args()
  
  # Convert argparse.Namespace to Config
  config = ModelConfig(
    data_path=args.data_path,
    train_split=args.train_split,
    validation_split=args.validation_split,
    model=args.model,
    batch_size=args.batch_size,
    burst=args.burst,
    channels=args.channels,
    epochs=args.epochs,
    lr=args.lr,
    optimizer=args.optimizer,
    loss=args.loss,
    model_save_path=args.model_save_path,
    log_dir=args.log_dir,
    seed=args.seed
  )
  
  return config


def _add_data_arguments(parser: argparse.ArgumentParser) -> None:
  """Add data-related arguments to parser."""
  data_group = parser.add_argument_group('Data Parameters')
  
  data_group.add_argument(
    '--data_path',
    type=str,
    required=True,
    help='Path to the dataset'
  )
  
  data_group.add_argument(
    '--train_split',
    type=float,
    default=0.8,
    help='Ratio of training data split'
  )
  
  data_group.add_argument(
    '--validation_split',
    type=float,
    default=0.0,
    help='Ratio of validation data split'
  )
  
  data_group.add_argument(
    '--model',
    type=str,
    default='grayscale',
    choices=['grayscale', 'timeless'],
    help='Feature matching method'
  )


def _add_training_arguments(parser: argparse.ArgumentParser) -> None:
  """Add training-related arguments to parser."""
  training_group = parser.add_argument_group('Training Parameters')
  
  training_group.add_argument(
    '--batch_size',
    type=int,
    default=32,
    help='Batch size for training'
  )
  training_group.add_argument(
    '--burst',
    type=int,
    default=3,
    help='Number of images per burst'
  )
  training_group.add_argument(
    '--channels',
    type=int,
    default=1,
    help='Number of channels per image'
  )
  
  training_group.add_argument(
    '--epochs',
    type=int,
    default=100,
    help='Number of training epochs'
  )
  
  training_group.add_argument(
    '--lr',
    type=float,
    default=0.001,
    help='Learning rate for optimizer'
  )
  
  training_group.add_argument(
    '--optimizer',
    type=str,
    default='adam',
    choices=['adam', 'sgd', 'rmsprop'],
    help='Optimizer for training'
  )
  
  training_group.add_argument(
    '--loss',
    type=str,
    default='quaternion',
    choices=['quaternion', 'angular', 'detailed', 'geodesic'],
    help='Loss function for training'
  )


def _add_output_arguments(parser: argparse.ArgumentParser) -> None:
  """Add output-related arguments to parser."""
  output_group = parser.add_argument_group('Output Parameters')
  
  output_group.add_argument(
    '--model_save_path',
    type=str,
    default=None,
    help='Path to save the trained model'
  )
  
  output_group.add_argument(
    '--log_dir',
    type=str,
    default='./logs',
    help='Directory for tensorboard logs'
  )


def _add_misc_arguments(parser: argparse.ArgumentParser) -> None:
  """Add miscellaneous arguments to parser."""
  misc_group = parser.add_argument_group('Miscellaneous Parameters')
  
  misc_group.add_argument(
    '--seed',
    type=int,
    default=42,
    help='Random seed for reproducibility'
  )