import os
import numpy as np
import tensorflow as tf

from datetime import datetime
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from models.light_glue_model import extract_points

class DataLoader:
  def __init__(self, data_path, train_split, validation_split, seed):
    """
    Initialize DataLoader with configuration

    Args:
      data_path: Path to the data directory
      train_split: Proportion of data to use for training
      validation_split: Proportion of data to use for validation
      seed: Random seed for data splitting
    """
    self.data_path = data_path
    self.train_split = train_split
    self.validation_split = validation_split
    self.seed = seed

  def load_data(self):
    """Load data based on file extension and type"""

    # Get all filenames in the folder
    files = os.listdir(self.data_path)

    return self._process_data(files)

  def _process_data(self, files):
    """
    Abstract method to process the loaded data.
    Must be implemented by subclasses.
    """
    raise NotImplementedError("Subclasses must implement this method")

  def _extract_data_from_filename(self, filename):
    """
    Extract timestamp, satellite position, and rotation from filename

    Args:
      filename: Name of the image file

    Returns:
      timestamp: Float representing the time
      sat_pos: Numpy array of satellite position
      sat_rot: Numpy array of satellite rotation (quaternion)
    """
    # Extract the relevant parts of the filename
    name_parts = filename.split('_')

    date_str = name_parts[1]
    sat_pos = np.array(list(map(float, name_parts[2].split(','))))
    sat_rot = np.array(list(map(float, name_parts[3].replace('.jpg', '').split(','))))

    # Convert date to timestamp
    date_time = datetime.strptime(date_str, "%d-%m-%Y %H:%M:%S.%f")
    timestamp = date_time.timestamp()  # This gives you a float representing the time

    return timestamp, sat_pos, sat_rot

  def _split_data(self, points, numerical_data, targets):
    """
    Split the data into training and validation sets

    Args:
      points: numpy array of data (e.g., matched keypoints or pixels)
      numerical_data: numpy array of numerical features (timestamps, satellite positions)
      targets: numpy array of target values (satellite rotations)

    Returns:
      Dictionary containing train and validation splits for all data types
    """
    # Use sklearn's train_test_split to split the data
    points_train, points_val, \
    num_train, num_val, \
    targets_train, targets_val = train_test_split(
      points,
      numerical_data,
      targets,
      test_size=1 - (self.train_split + self.validation_split),
      random_state=self.seed,
    )

    # Return dictionary containing all splits
    return {
      'train': {
        'image_data': points_train,
        'numerical': num_train,
        'targets': targets_train
      },
      'val': {
        'image_data': points_val,
        'numerical': num_val,
        'targets': targets_val
      }
    }

# Subclass for feature matching between image pairs
class MatchingDataLoader(DataLoader):
  def __init__(self, data_path, train_split, validation_split, seed, matching_method, num_matches):
    assert matching_method is not None, "matching_method must be specified for MatchingDataLoader"
    assert num_matches is not None, "num_matches must be specified for MatchingDataLoader"
    super().__init__(data_path, train_split, validation_split, seed)
    self.matching_method = matching_method
    self.num_matches = num_matches

  def _process_data(self, files):
    """
    Process data by matching features between consecutive images

    Args:
      files: List of filenames in the data directory

    Returns:
      Processed and split data for training
    """
    points = []
    numerical_data = []
    targets = []

    for i in range(len(files)-1)[:10]:
      print(f"Processing image pair {i + 1}/{len(files) - 1}")
      filename0 = files[i]
      filename1 = files[i + 1]

      timestamp0, sat_pos0, sat_rot0 = self._extract_data_from_filename(filename0)
      timestamp1, sat_pos1, sat_rot1 = self._extract_data_from_filename(filename1)

      # coordinates with shape (K, 2)
      points0, points1 = self._matching_features(filename0, filename1)

      # Get the number of matches K
      K = points0.shape[0]

      # Check if there are no matches
      #if K == 0:
      #  print(f"No matches found between images {i} and {i + 1}")
      #  continue

      if K >= self.num_matches:
        # Keep only the first N matches
        points0 = points0[:self.num_matches]
        points1 = points1[:self.num_matches]
      else:
        # Pad with dummy matches if there are fewer than N matches
        num_missing = self.num_matches - K
        # Create dummy matches with zeros
        dummy_points0 = tf.zeros((num_missing, 2), dtype=tf.float32)
        dummy_points1 = tf.zeros((num_missing, 2), dtype=tf.float32)
        # Concatenate the real and dummy matches
        points0 = tf.concat([points0.cpu().numpy(), dummy_points0], axis=0)
        points1 = tf.concat([points1.cpu().numpy(), dummy_points1], axis=0)

      # This will result in shape (2N, 2)
      points.append(np.concatenate((
        points0.numpy().reshape(-1),
        points1.numpy().reshape(-1)
      )))

      # Concatenate timestamp and satellite position
      numerical_data.append(
        np.concatenate([
          np.array([timestamp0]),  # Convert scalar to 1D array
          np.array([timestamp1]),
          sat_pos0,
          sat_pos1
        ], axis=0)
      )
      # Satellite rotation is the target, already normalized quaternion
      targets.append(np.concatenate([sat_rot0, sat_rot1], axis=0))

    # Convert lists to numpy arrays
    numerical_data = np.array(numerical_data)
    targets = np.array(targets)
    points = np.array(points)

    # Split data
    return self._split_data(points, numerical_data, targets)

  def _matching_features(self, img0, img1):
    """
    Perform feature matching between two images

    Args:
      img0: Filename of the first image
      img1: Filename of the second image

    Returns:
      points0: Keypoints from the first image
      points1: Corresponding keypoints from the second image
    """
    image0_path = os.path.join(self.data_path, img0)
    image1_path = os.path.join(self.data_path, img1)

    # Implement your matching method here
    # For example, using LightGlue or any other method
    if self.matching_method == 'light_glue':
      return extract_points(image0_path, image1_path)
    else:
      raise NotImplementedError(f"Matching method '{self.matching_method}' is not implemented")

# Subclass for processing individual images and extracting grayscale pixels
class GrayscaleDataLoader(DataLoader):
  def _process_data(self, files):
    """
    Process data by converting images to grayscale and extracting pixels

    Args:
      files: List of filenames in the data directory

    Returns:
      Processed and split data for training
    """
    image_data = []
    numerical_data = []
    targets = []

    for i in range(len(files)):
      print(f"Processing image {i + 1}/{len(files)}")
      filename = files[i]

      # Extract data from filename
      timestamp, sat_pos, sat_rot = self._extract_data_from_filename(filename)

      # Load image and get grayscale pixels
      pixels = self._get_grayscale_pixels(filename)

      # Append data to lists
      image_data.append(pixels)
      numerical_data.append(
        np.concatenate([np.array([timestamp]), sat_pos], axis=0)
      )
      targets.append(sat_rot)

    # Convert lists to numpy arrays
    numerical_data = np.array(numerical_data)
    targets = np.array(targets)
    image_data = np.array(image_data)

    # Split data
    return self._split_data(image_data, numerical_data, targets)

  def _get_grayscale_pixels(self, filename):
    """
    Load an image, convert it to grayscale, normalize, and flatten the pixel values

    Args:
      filename: Name of the image file

    Returns:
      Flattened numpy array of grayscale pixel values
    """
    image_path = os.path.join(self.data_path, filename)
    # Load the image in grayscale
    img = load_img(image_path, color_mode='grayscale')
    # Convert image to array and normalize
    img_array = img_to_array(img) / 255.0
    return img_array