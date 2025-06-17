import os
import tensorflow as tf

def setup_environment() -> None:
  """Set up environment variables for TensorFlow and CUDA."""
  os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Show only errors
  cuda_env_path = os.path.abspath('../micromamba/envs/thesis')
  os.environ['XLA_FLAGS'] = f'--xla_gpu_cuda_data_dir={cuda_env_path}'
  os.environ['LD_LIBRARY_PATH'] = f'{cuda_env_path}/lib:' + os.environ.get('LD_LIBRARY_PATH', '')
  os.environ['PATH'] = f'{cuda_env_path}/bin:' + os.environ.get('PATH', '')

  _setup_gpu()

def _setup_gpu() -> None:
  """Configure GPU settings."""
  gpus = tf.config.experimental.list_physical_devices('GPU')
  if gpus:
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)