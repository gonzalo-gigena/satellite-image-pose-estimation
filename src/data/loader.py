import os
import time as ti
from typing import List, Tuple, TypeAlias, TypedDict

import numpy as np
from numpy.typing import NDArray
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tqdm import tqdm

from config.model_config import ModelConfig

FileMetadata: TypeAlias = Tuple[
    str,  # sat_index
    str,  # num_bursts
    str,  # burst_index
    int,  # elapsed_time
    NDArray[np.floating],  # sat_position (3D)
    NDArray[np.floating],  # sat_rotation (4D quaternion)
]


class DataSplit(TypedDict):
  image_data: NDArray[np.floating]
  numerical: NDArray[np.floating]
  targets: NDArray[np.floating]


class TrainValData(TypedDict):
  train: DataSplit
  val: DataSplit


class BaseDataLoader:
  """Base class for loading and processing data."""

  def __init__(self, config: ModelConfig) -> None:
    """
    Initialize DataLoader with configuration.

    Args:
      data_path: Path to the data directory
      train_split: Proportion of data to use for training
      validation_split: Proportion of data to use for validation
      seed: Random seed for data splitting
      channels: Number of image channels (1 for grayscale, 3 for RGB)
      burst: Number of burst images
    """
    self.data_path = config.data_path
    self.train_split = config.train_split
    self.validation_split = config.validation_split
    self.seed = config.seed
    self.channels = config.channels
    self.frames = config.frames

  def load_data(self) -> TrainValData:
    """Load data based on file extension and type.

    Returns:
      Dictionary containing train and validation data splits
    """
    # Get all filenames in the folder
    files: List[str] = [f for f in os.listdir(self.data_path) if f.startswith('cubesat')]
    files.sort()  # The order of files is importat for loading the images
    return self._process_data(files)

  def _process_data(self, files: List[str]) -> TrainValData:
    """
    Abstract method to process the loaded data.
    Must be implemented by subclasses.

    Args:
      files: List of filenames to process

    Returns:
      Processed data dictionary
    """
    raise NotImplementedError('Subclasses must implement this method')

  def _extract_data_from_filename(self, filename: str) -> FileMetadata:
    """
    Extract timestamp, satellite position, and rotation from filename.

    Args:
      filename: Name of the image file

    Returns:
      Tuple containing:
        - i: Satellite index
        - j: Number of bursts
        - k: Burst index
        - elapsed_time: Time elapsed
        - sat_pos: Numpy array of satellite position
        - sat_rot: Numpy array of satellite rotation (quaternion)
    """
    # Extract the relevant parts of the filename
    # filePath =
    # $"{screenshotFolder}/{sat.name}_{sat.index}_{sat.numBurst}_{sat.burstIndex}_{sat.time}_{satPos}_{satRot}.jpg";
    file_name_parts: List[str] = filename.split('_')

    _: str = file_name_parts[0]  # satellite name
    i: str = file_name_parts[1]
    j: str = file_name_parts[2]
    k: str = file_name_parts[3]
    elapsed_time: int = int(file_name_parts[4])
    sat_pos: NDArray[np.floating] = np.array(list(map(float, file_name_parts[5].split(','))))
    sat_rot: NDArray[np.floating] = np.array(list(map(float, file_name_parts[6].replace('.jpg', '').split(','))))

    # TODO: normalize elapsed_time [0, 1)
    return i, j, k, elapsed_time, sat_pos, sat_rot

  def _get_pixels(self, filename: str) -> NDArray[np.floating]:
    """
    Load an image, normalize, and return the pixel values.

    Args:
      filename: Name of the image file

    Returns:
      Normalized image array
    """
    image_path: str = os.path.join(self.data_path, filename)

    # Load the image
    if self.channels == 1:
      img = load_img(image_path, color_mode='grayscale')
    else:
      img = load_img(image_path)

    # Convert image to array and normalize
    img_array: NDArray[np.floating] = img_to_array(img) / 255.0
    return img_array

  def _split_data(
      self, images: NDArray[np.floating], numerical_data: NDArray[np.floating], targets: NDArray[np.floating]
  ) -> TrainValData:
    """
    Split the data into training and validation sets.

    Args:
      images: Numpy array of data
      numerical_data: Numpy array of numerical features (timestamps, satellite positions)
      targets: Numpy array of target values (satellite rotations)

    Returns:
      Dictionary containing train and validation splits for all data types
    """
    # Use sklearn's train_test_split to split the data
    images_train, images_val, num_train, num_val, targets_train, targets_val = train_test_split(
        images,
        numerical_data,
        targets,
        test_size=1 - (self.train_split + self.validation_split),
        random_state=self.seed,
    )

    # Return dictionary containing all splits
    return {
        'train': {'image_data': images_train, 'numerical': num_train, 'targets': targets_train},
        'val': {'image_data': images_val, 'numerical': num_val, 'targets': targets_val},
    }


class DataLoader(BaseDataLoader):
  def _process_data(self, files: List[str]) -> TrainValData:
    """
    Process data by converting images to grayscale and extracting pixels

    Args:
      files: List of filenames in the data directory

    Returns:
      Processed and split data for training
    """
    images: List[List[NDArray[np.floating]]] = []
    time: List[str] = []
    positions: List[NDArray[np.floating]] = []
    targets: List[NDArray[np.floating]] = []

    start_time = ti.time()
    # Create progress bar for sequences
    print(f'Processing {len(files)} images ...')
    num_sequences = len(files) // self.frames
    with tqdm(total=num_sequences, desc='Processing sequences', unit='seq') as pbar:
      for i in range(0, len(files), self.frames):
        files_data: List[FileMetadata] = []
        for j in range(self.frames):
          files_data.append(self._extract_data_from_filename(files[i + j]))

        if not self._validate(files_data):
          pbar.set_postfix_str('Skipped invalid sequence')
          pbar.update(1)
          continue

        # Load image and get grayscale pixels
        pixels: List[NDArray[np.floating]] = []
        for j in range(self.frames):
          pixels.append(self._get_pixels(files[i + j]))

        # Append data to lists
        images.append(pixels)
        time.append(files_data[-1][-3])  # time elapsed of last image
        positions.append(files_data[-1][-2])  # sat position of last image
        targets.append(files_data[-1][-1])  # sat rotation of last image

        pbar.update(1)

    # Convert Python lists to NumPy arrays
    images_array: NDArray[np.floating] = np.array(images)  # shape: (N, B, H, W, C)
    time_array: NDArray[np.floating] = np.array(time, dtype=np.float32)  # shape: (N,)
    positions_array: NDArray[np.floating] = np.array(positions)  # shape: (N, 3)
    targets_array: NDArray[np.floating] = np.array(targets)  # shape: (N, 4)

    # Combine time and positions, then normalize together
    numerical_data: NDArray[np.floating] = np.column_stack([time_array, positions_array])
    scaler: StandardScaler = StandardScaler()
    numerical_data_norm: NDArray[np.floating] = scaler.fit_transform(numerical_data)

    end_time = ti.time()
    execution_time = end_time - start_time
    print(f'Processing time: {execution_time:.4f} seconds')

    # (N, 3, 102, 102, 1) (N, 4) (N, 4)
    return self._split_data(images_array, numerical_data_norm, targets_array)

  def _validate(self, data: List[FileMetadata]) -> bool:
    """
    Validate that all files in a sequence belong to the same position and burst.

    Args:
      data: List of extracted file data tuples

    Returns:
      True if validation passes, False otherwise
    """
    for i in range(len(data) - 1):
      if data[i][0] != data[i + 1][0]:
        return False  # Make sure images are in the same position
      if data[i][1] != data[i + 1][1]:
        return False  # Make sure images are in the same burst
    return True
