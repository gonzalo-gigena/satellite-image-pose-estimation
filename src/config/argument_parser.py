import argparse

from .model_config import ModelConfig


def parse_args() -> ModelConfig:
  """Parse command line arguments and return ModelConfig object."""
  parser = argparse.ArgumentParser(
      description='Machine Learning Model Parameters', formatter_class=argparse.ArgumentDefaultsHelpFormatter
  )

  _add_data_arguments(parser)

  _add_training_arguments(parser)

  _add_output_arguments(parser)

  _add_misc_arguments(parser)

  args = parser.parse_args()

  # Convert argparse.Namespace to Config
  config = ModelConfig(
      data_path=args.data_path,
      train_split=args.train_split,
      validation_split=args.validation_split,
      model=args.model,
      load_weights=args.load_weights,
      branch_type=args.branch_type,
      batch_size=args.batch_size,
      frames=args.frames,
      channels=args.channels,
      image_height=args.image_height,
      image_width=args.image_width,
      epochs=args.epochs,
      lr=args.lr,
      optimizer=args.optimizer,
      loss=args.loss,
      log_dir=args.log_dir,
      seed=args.seed,
      resume_training=args.resume_training
  )

  return config


def _add_data_arguments(parser: argparse.ArgumentParser) -> None:
  """Add data-related arguments to parser."""
  data_group = parser.add_argument_group('Data Parameters')

  data_group.add_argument('-d', '--data_path', type=str, required=True, help='Path to the dataset')

  data_group.add_argument('-t', '--train_split', type=float, default=0.8, help='Ratio of training data split')

  data_group.add_argument('-v', '--validation_split', type=float, default=0.0, help='Ratio of validation data split')

  data_group.add_argument(
      '-m', '--model', type=str, default='relative_pose', choices=['relative_pose'], help='Feature matching method'
  )

  data_group.add_argument(
      '-bt', '--branch_type',
      type=str,
      default=None,
      choices=[
          'cnnA',
          'cnnAspp',
          'cnnB',
          'cnnBspp'],
      help='Feature matching method')


def _add_training_arguments(parser: argparse.ArgumentParser) -> None:
  """Add training-related arguments to parser."""
  training_group = parser.add_argument_group('Training Parameters')

  training_group.add_argument('-lw', '--load_weights', action='store_true')

  training_group.add_argument('-rt', '--resume_training', action='store_true')

  training_group.add_argument('-ih', '--image_height', type=int, default=102, help='Image height')

  training_group.add_argument('-iw', '--image_width', type=int, default=102, help='Image width')

  training_group.add_argument('-b', '--batch_size', type=int, default=32, help='Batch size for training')

  training_group.add_argument('-f', '--frames', type=int, default=3, help='Number of frames per burst')

  training_group.add_argument('-c', '--channels', type=int, default=1, help='Number of channels per image')

  training_group.add_argument('-e', '--epochs', type=int, default=100, help='Number of training epochs')

  training_group.add_argument('-lr', '--lr', type=float, default=0.001, help='Learning rate for optimizer')

  training_group.add_argument(
      '-o', '--optimizer', type=str, default='adam', choices=['adam', 'sgd', 'rmsprop'], help='Optimizer for training'
  )

  training_group.add_argument(
      '-l', '--loss',
      type=str,
      default='quaternion',
      choices=['quaternion', 'angular', 'detailed', 'geodesic'],
      help='Loss function for training',
  )


def _add_output_arguments(parser: argparse.ArgumentParser) -> None:
  """Add output-related arguments to parser."""
  output_group = parser.add_argument_group('Output Parameters')

  output_group.add_argument('-ld', '--log_dir', type=str, default='./logs', help='Directory for tensorboard logs')


def _add_misc_arguments(parser: argparse.ArgumentParser) -> None:
  """Add miscellaneous arguments to parser."""
  misc_group = parser.add_argument_group('Miscellaneous Parameters')

  misc_group.add_argument('-s', '--seed', type=int, default=42, help='Random seed for reproducibility')
