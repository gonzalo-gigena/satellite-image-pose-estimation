import os
import argparse
import tensorflow as tf

from main import main

def setup_environment():
  """Set up environment variables for TensorFlow and CUDA."""
  os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Show only errors
  cuda_env_path = os.path.abspath('../micromamba/envs/thesis')
  os.environ['XLA_FLAGS'] = f'--xla_gpu_cuda_data_dir={cuda_env_path}'
  os.environ['LD_LIBRARY_PATH'] = f'{cuda_env_path}/lib:' + os.environ.get('LD_LIBRARY_PATH', '')
  os.environ['PATH'] = f'{cuda_env_path}/bin:' + os.environ.get('PATH', '')

  gpus = tf.config.experimental.list_physical_devices('GPU')
  if gpus:
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)

def parse_args():
  """Parse command line arguments for ML model training."""
  parser = argparse.ArgumentParser(
      description='Machine Learning Model Parameters',
      formatter_class=argparse.ArgumentDefaultsHelpFormatter
  )

  # Data parameters
  parser.add_argument('--data_path', type=str, required=True,
                      help='Path to the dataset')
  parser.add_argument('--train_split', type=float, default=0.8,
                      help='Ratio of training data split')
  parser.add_argument('--validation_split', type=float, default=0.0,
                      help='Ratio of validation data split')
  parser.add_argument('--model', type=str, default='grayscale',
                      choices=['grayscale', 'light_glue'],
                      help='Feature matching method')
  parser.add_argument('--num_matches', type=int, default=None,
                      help='desired fixed number of matches')
  
  # Training parameters
  parser.add_argument('--batch_size', type=int, default=32,
                      help='Batch size for training')
  parser.add_argument('--epochs', type=int, default=100,
                      help='Number of training epochs')
  parser.add_argument('--lr', type=float, default=0.001,
                      help='Learning rate for optimizer')
  parser.add_argument('--optimizer', type=str, default='adam',
                      choices=['adam', 'sgd', 'rmsprop'],
                      help='Optimizer for training')
  parser.add_argument('--loss', type=str, default='quaternion',
                      choices=['quaternion', 'angular', 'detailed'],
                      help='Loss function for training')
  
  # Output parameters
  parser.add_argument('--model_save_path', type=str, default=None,
                      help='Path to save the trained model')
  parser.add_argument('--log_dir', type=str, default='./logs',
                      help='Directory for tensorboard logs')
  
  # Misc parameters
  parser.add_argument('--seed', type=int, default=42,
                      help='Random seed for reproducibility')

  args = parser.parse_args()
  
  # Argument validation
  if not 0 <= args.train_split <= 1:
    parser.error('Train split must be between 0 and 1.')
  if not 0 <= args.validation_split <= 1:
    parser.error('Validation split must be between 0 and 1.')

  # Handle dependencies between arguments
  if args.matching_method == 'light_glue':
    if args.num_matches is None:
      parser.error('--num_matches is required when --model is "light_glue".')
  else:
    if args.num_matches is not None:
      parser.error('--num_matches should not be set when --model is not "light_glue".')
  
  return args

if __name__ == '__main__':
  setup_environment()
  args = parse_args()
  main(args)