import tensorflow as tf

from data_loader import MatchingDataLoader, GrayscaleDataLoader
from models.feature_matching_model import (
  DataGenerator as MatchingDataGenerator,
  FeatureMatchingModel
)
from models.grayscale_model import (
  DataGenerator as GrayscaleDataGenerator,
  GrayscaleModel
)

from models.loss import quaternion_loss, angular_distance_loss, detailed_distance_loss

def get_loss_function(loss_name):
  """Select and return the loss function based on the given name"""
  loss_functions = {
    'quaternion': quaternion_loss,
    'angular': angular_distance_loss,
    'detailed': detailed_distance_loss,
  }

  loss_function = loss_functions.get(loss_name.lower())
  if loss_function is None:
    raise ValueError(f"Unsupported loss function: {loss_function}")

  return loss_function

def get_optimizer(optimizer_name, learning_rate):
  """Select and return the optimizer based on the given name."""
  optimizers = {
    'adam': tf.keras.optimizers.Adam,
    'sgd': tf.keras.optimizers.SGD,
    'rmsprop': tf.keras.optimizers.RMSprop,
  }

  optimizer_class = optimizers.get(optimizer_name.lower())
  if optimizer_class is None:
    raise ValueError(f"Unsupported optimizer: {optimizer_name}")

  return optimizer_class(learning_rate=learning_rate)



def get_model(model):
  """Select and return the appropriate model based on the matching method.
  
  Args:
      matching_method (str, optional): The matching method to use. 
          If None, returns GrayscaleModel.
  
  Returns:
      tf.keras.Model: The selected model instance.
  """
  if model == 'light_glue':
    return FeatureMatchingModel()
  return GrayscaleModel()

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
      batch_size=batch_size
    )
  return GrayscaleDataGenerator(
    images=data['image_data'],
    numerical=data['numerical'],
    targets=data['targets'],
    shuffle=shuffle,
    batch_size=batch_size,
    augment=augment
  )