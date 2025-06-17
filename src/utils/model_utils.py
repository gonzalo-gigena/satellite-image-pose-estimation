from typing import Union, Dict, Any
from config.model_config import ModelConfig

from models.timeless_model import TimelessModel, TimelessDataLoader
from models.grayscale_model import GrayscaleDataLoader, GrayscaleModel, GrayscaleDataGenerator
from models.data_loader import DataSplit

def get_model(model: str, channels: int) -> Union[TimelessModel, GrayscaleModel]:
  """Select and return the appropriate model."""
  if model == 'timeless':
    return TimelessModel()
  return GrayscaleModel(channels=channels)


def get_data_loader(config: ModelConfig) -> Union[TimelessDataLoader, GrayscaleDataLoader]:
  """Select and return the appropriate data loader based on the matching method.
  
  Args:
    config: ModelConfig object containing all configuration parameters
  
  Returns:
    DataLoader: The appropriate data loader instance
  """
  if config.model == 'timeless':
    return TimelessDataLoader(config)
  return GrayscaleDataLoader(config)


def get_train_generator(
  data: DataSplit, 
  batch_size: int, 
  model: str, 
  shuffle: bool = True, 
  augment: bool = False
) -> GrayscaleDataGenerator:
  """Create and return the appropriate data generator based on the matching method.
  
  Args:
    data: Dictionary containing the training data with keys 'image_data', 'numerical', 'targets'
    batch_size: Batch size for training
    model: Model type identifier (currently unused but kept for future extensibility)
    shuffle: Whether to shuffle the data
    augment: Whether to apply data augmentation
  
  Returns:
    GrayscaleDataGenerator: The data generator instance
  """
  return GrayscaleDataGenerator(
    images=data['image_data'],
    numerical=data['numerical'],
    targets=data['targets'],
    shuffle=shuffle,
    batch_size=batch_size,
    augment=augment
  )