import argparse

def parse_args():
  """Parse command line arguments for ML model training."""
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
  
  # Misc parameters
  _add_misc_arguments(parser)

  args = parser.parse_args()
  _validate_arguments(args, parser)
  
  return args

def _add_data_arguments(parser):
  """Add data-related arguments to parser."""
  data_group = parser.add_argument_group('Data Parameters')
  data_group.add_argument('--data_path', type=str, required=True,
                        help='Path to the dataset')
  data_group.add_argument('--train_split', type=float, default=0.8,
                        help='Ratio of training data split')
  data_group.add_argument('--validation_split', type=float, default=0.0,
                        help='Ratio of validation data split')
  data_group.add_argument('--model', type=str, default='grayscale',
                        choices=['grayscale', 'light_glue', 'timeless', 'sequential'],
                        help='Feature matching method')
  data_group.add_argument('--num_matches', type=int, default=None,
                        help='desired fixed number of matches')

def _add_training_arguments(parser):
  """Add training-related arguments to parser."""
  training_group = parser.add_argument_group('Training Parameters')
  training_group.add_argument('--batch_size', type=int, default=32,
                            help='Batch size for training')
  training_group.add_argument('--epochs', type=int, default=100,
                            help='Number of training epochs')
  training_group.add_argument('--lr', type=float, default=0.001,
                            help='Learning rate for optimizer')
  training_group.add_argument('--optimizer', type=str, default='adam',
                            choices=['adam', 'sgd', 'rmsprop'],
                            help='Optimizer for training')
  training_group.add_argument('--loss', type=str, default='quaternion',
                            choices=['quaternion', 'angular', 'detailed'],
                            help='Loss function for training')

def _add_output_arguments(parser):
  """Add output-related arguments to parser."""
  output_group = parser.add_argument_group('Output Parameters')
  output_group.add_argument('--model_save_path', type=str, default=None,
                          help='Path to save the trained model')
  output_group.add_argument('--log_dir', type=str, default='./logs',
                          help='Directory for tensorboard logs')

def _add_misc_arguments(parser):
  """Add miscellaneous arguments to parser."""
  misc_group = parser.add_argument_group('Miscellaneous Parameters')
  misc_group.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')

def _validate_arguments(args, parser):
  """Validate parsed arguments."""
  if not 0 <= args.train_split <= 1:
    parser.error('Train split must be between 0 and 1.')
  if not 0 <= args.validation_split <= 1:
    parser.error('Validation split must be between 0 and 1.')

  # Handle dependencies between arguments
  if args.model == 'light_glue':
    if args.num_matches is None:
      parser.error('--num_matches is required when --model is "light_glue".')
  else:
    if args.num_matches is not None:
      parser.error('--num_matches should not be set when --model is not "light_glue".')
