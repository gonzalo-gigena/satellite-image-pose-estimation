import argparse
from dataclasses import fields

from .model_config import ModelConfig


def parse_args() -> ModelConfig:
  """Parse command line arguments and return ModelConfig object."""
  parser = argparse.ArgumentParser(
      description='Machine Learning Model Parameters',
      formatter_class=argparse.ArgumentDefaultsHelpFormatter
  )

  _add_data_arguments(parser)
  _add_training_arguments(parser)
  _add_output_arguments(parser)
  _add_misc_arguments(parser)

  args = parser.parse_args()

  # Convert argparse.Namespace to ModelConfig using dataclass fields
  # This automatically maps argument names to config fields
  config_dict = {
      field.name: getattr(args, field.name)
      for field in fields(ModelConfig)
      if hasattr(args, field.name) and getattr(args, field.name) is not None
  }

  return ModelConfig(**config_dict)


def _add_data_arguments(parser: argparse.ArgumentParser) -> None:
  """Add data-related arguments to parser."""
  data_group = parser.add_argument_group('Data Parameters')

  data_group.add_argument('-d', '--data_path', type=str, required=True, help='Path to the dataset')

  data_group.add_argument(
      '-t', '--train_split', type=float,
      help='Ratio of training data split (must be between 0 and 1, and train_split + validation_split <= 1)'
  )

  data_group.add_argument(
      '-v', '--validation_split', type=float,
      help='Ratio of validation data split (must be between 0 and 1, and train_split + validation_split <= 1)'
  )


def _add_training_arguments(parser: argparse.ArgumentParser) -> None:
  """Add training-related arguments to parser."""
  training_group = parser.add_argument_group('Training Parameters')

  training_group.add_argument(
      '-lw', '--load_weights', action='store_true',
      help='Load pre-trained weights'
  )

  training_group.add_argument(
      '-tw', '--train_weights', action='store_true',
      help='Train loaded weights (requires --load_weights)'
  )

  training_group.add_argument(
      '-rt', '--resume_training', action='store_true',
      help='Resume training from checkpoint'
  )

  training_group.add_argument('-ih', '--image_height', type=int, help='Image height')

  training_group.add_argument('-iw', '--image_width', type=int, help='Image width')

  training_group.add_argument('-b', '--batch_size', type=int, help='Batch size for training')

  training_group.add_argument('-f', '--frames', type=int, help='Number of frames per burst')

  training_group.add_argument('-c', '--channels', type=int, help='Number of channels per image')

  training_group.add_argument('-e', '--epochs', type=int, help='Number of training epochs')

  training_group.add_argument('-lr', '--lr', type=float, help='Learning rate for optimizer')

  training_group.add_argument(
      '-o', '--optimizer', type=str, choices=['adam', 'sgd', 'rmsprop'], help='Optimizer for training'
  )

  training_group.add_argument(
      '-bt', '--branch_type',
      type=str,
      choices=['cnnA', 'cnnAspp', 'cnnB', 'cnnBspp'],
      help='Branch type for relative_pose model (defaults to cnnAspp if not specified)')

  training_group.add_argument(
      '-l', '--loss',
      type=str,
      choices=['quaternion', 'angular', 'detailed', 'geodesic'],
      help='Loss function for training',
  )


def _add_output_arguments(parser: argparse.ArgumentParser) -> None:
  """Add output-related arguments to parser."""
  output_group = parser.add_argument_group('Output Parameters')

  output_group.add_argument('-ld', '--log_dir', type=str, help='Directory for tensorboard logs')


def _add_misc_arguments(parser: argparse.ArgumentParser) -> None:
  """Add miscellaneous arguments to parser."""
  misc_group = parser.add_argument_group('Miscellaneous Parameters')

  misc_group.add_argument('-s', '--seed', type=int, help='Random seed for reproducibility')
