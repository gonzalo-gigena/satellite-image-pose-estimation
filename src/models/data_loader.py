import os
import numpy as np

from datetime import datetime
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import load_img, img_to_array

class DataLoader:
  def __init__(self, data_path, train_split, validation_split, seed, channels=1):
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
    self.channels = channels

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

  def _get_pixels(self, filename):
    """
    Load an image, normalize, and flatten the pixel values

    Args:
      filename: Name of the image file

    Returns:
      Flattened numpy array
    """
    image_path = os.path.join(self.data_path, filename)
    # Load the image
    img = load_img(image_path, color_mode='grayscale') if self.channels == 1 else load_img(image_path)
    # Convert image to array and normalize
    img_array = img_to_array(img) / 255.0
    return img_array

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