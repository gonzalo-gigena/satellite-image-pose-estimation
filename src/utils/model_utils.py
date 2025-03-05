from models.feature_matching_model import (
  FeatureMatchingModel, MatchingDataGenerator, MatchingDataLoader
)
from models.grayscale_model import (
  GrayscaleModel,
  ImprovedGrayscaleModel,
  TransferLearningGrayscaleModel,
  GrayscaleDataGenerator,
  GrayscaleDataLoader
)
from models.timeless_model import TimelessModel, TimelessDataLoader

def get_model(model):
  """Select and return the appropriate model."""
  if model == 'light_glue':
    return FeatureMatchingModel()
  if model == 'timeless':
    return TimelessModel()
  return ImprovedGrayscaleModel()

def get_data_loader(data_path, train_split, validation_split, seed, 
                   model, num_matches):
  """Select and return the appropriate data loader based on the matching method.
  
  Args:
    data_path (str): Path to the data directory
    train_split (float): Proportion of data to use for training
    validation_split (float): Proportion of data to use for validation
    seed (int): Random seed for reproducibility
    matching_method (str): The model to use
    num_matches (int, optional): Number of matches to use
  
  Returns:
    DataLoader: The appropriate data loader instance
  """
  if model == 'light_glue':
    return MatchingDataLoader(
      data_path=data_path,
      train_split=train_split,
      validation_split=validation_split,
      seed=seed,
      matching_method=model,
      num_matches=num_matches
    )
  if model == 'timeless':
    return TimelessDataLoader(
      data_path=data_path,
      train_split=train_split,
      validation_split=validation_split,
      seed=seed
    )
  return GrayscaleDataLoader(
    data_path=data_path,
    train_split=train_split,
    validation_split=validation_split,
    seed=seed
  )

def get_train_generator(data, batch_size, model, shuffle=True,  augment=False):
  """Create and return the appropriate data generator based on the matching method.
  
  Args:
    data (dict): Dictionary containing the training data
    batch_size (int): Batch size for training
    matching_method (str, optional): The matching method to use
    shuffle (bool): Whether to shuffle the data
  
  Returns:
    DataGenerator: The appropriate data generator instance
  """
  if model == 'light_glue':
    return MatchingDataGenerator(
      points=data['image_data'],
      numerical=data['numerical'],
      targets=data['targets'],
      shuffle=shuffle,
      batch_size=batch_size,
      augment=augment
    )
  return GrayscaleDataGenerator(
    images=data['image_data'],
    numerical=data['numerical'],
    targets=data['targets'],
    shuffle=shuffle,
    batch_size=batch_size,
    augment=augment
  )