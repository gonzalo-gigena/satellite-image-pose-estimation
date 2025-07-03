import os


def setup_environment() -> None:
  """Set up environment variables for TensorFlow and CUDA."""
  os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # 0=all, 1=info, 2=warning, 3=error
  cuda_env_path = os.path.abspath("../micromamba/envs/thesis")
  os.environ["XLA_FLAGS"] = f"--xla_gpu_cuda_data_dir={cuda_env_path}"
  os.environ["LD_LIBRARY_PATH"] = f"{cuda_env_path}/lib:" + os.environ.get("LD_LIBRARY_PATH", "")
  os.environ["PATH"] = f"{cuda_env_path}/bin:" + os.environ.get("PATH", "")

  _setup_gpu()


def _setup_gpu() -> None:
  """Configure GPU settings."""
  import tensorflow as tf
  from tensorflow.python.platform import build_info

  gpus = tf.config.experimental.list_physical_devices("GPU")
  if gpus:
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)

  print("TensorFlow version:", tf.__version__)
  print("CUDA available:", tf.test.is_built_with_cuda())
  print("GPU available:", tf.config.list_physical_devices("GPU"))

  # Check built-in CUDA version
  print("Built with CUDA:", build_info.build_info["cuda_version"])
  print("Built with cuDNN:", build_info.build_info["cudnn_version"])
