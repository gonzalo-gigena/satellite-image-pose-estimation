import os
import time as ti
from dataclasses import dataclass
from typing import List, Tuple, TypedDict

import numpy as np
from numpy.typing import NDArray
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.preprocessing.image import img_to_array, load_img

from config.model_config import ModelConfig


@dataclass
class FileMetadata:
  sat_name: str
  sat_index: str
  elapsed_time: int
  sat_position: NDArray[np.floating]   # shape (3,)
  sat_rotation: NDArray[np.floating]   # shape (4,)


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
    """
    self.image_height = config.image_height
    self.image_width = config.image_width
    self.data_path = config.data_path
    self.train_split = config.train_split
    self.validation_split = config.validation_split
    self.seed = config.seed
    self.channels = config.channels
    self.frames = config.frames

  def load_files(self) -> None:
    files: List[str] = [f for f in os.listdir(self.data_path) if f.startswith('cubesat')]
    files.sort()  # The order of files is important for loading the images
    return files

  def load_data(self, files_chunk) -> TrainValData:
    """
    Load data based on file extension and type.

    Returns:
      Dictionary containing train and validation data splits
    """
    return self._process_data(files_chunk)

  def _process_data(self, files_chunk: List[str]) -> TrainValData:
    """
    Abstract method to process the loaded data.
    Must be implemented by subclasses.
    """
    raise NotImplementedError('Subclasses must implement this method')

  def _extract_metadata_from_filename(self, filename: str) -> FileMetadata:
    """
    Extract timestamp, satellite position, and rotation from filename.

    Filename format:
      <sat_name>_<sat_index>_<time>_<pos>_<rot>.jpg
    """
    file_name_parts: List[str] = filename.split('_')
    if len(file_name_parts) < 5:
      raise ValueError(f'Unexpected filename format: {filename}')

    sat_pos: NDArray[np.floating] = np.array(
        list(map(float, file_name_parts[3].split(','))), dtype=np.float32
    )
    sat_rot: NDArray[np.floating] = np.array(
        list(map(float, file_name_parts[4].replace('.jpg', '').split(','))),
        dtype=np.float32,
    )

    return FileMetadata(
        sat_name=file_name_parts[0],
        sat_index=file_name_parts[1],
        elapsed_time=int(file_name_parts[2]),
        sat_position=sat_pos,
        sat_rotation=sat_rot,
    )

  def _validate(self, data: List[FileMetadata]) -> bool:
    """
    Validate that all files in a sequence belong to the same sat_index and burst.
    """
    if not data:
      return False

    first_index = data[0].sat_index

    return all(d.sat_index == first_index for d in data)

  def _get_pixels(self, filename: str) -> NDArray[np.floating]:
    """
    Load an image, normalize, and return the pixel values.
    """
    image_path: str = os.path.join(self.data_path, filename)

    if self.channels == 1:
      img = load_img(image_path, color_mode='grayscale')
    else:
      img = load_img(image_path)

    img_array: NDArray[np.floating] = img_to_array(img) / 255.0

    if img_array.shape[:2] != (self.image_height, self.image_width):
      raise ValueError(
          f'Image {filename} has incorrect dimensions {img_array.shape[:2]}, '
          f'expected ({self.image_height}, {self.image_width})'
      )

    return img_array

  def _split_data(
      self,
      images: NDArray[np.floating],
      numerical_data: NDArray[np.floating],
      targets: NDArray[np.floating]
  ) -> TrainValData:
    """
    Split the data into training and validation sets.
    """
    images_train, images_val, num_train, num_val, targets_train, targets_val = train_test_split(
        images,
        numerical_data,
        targets,
        test_size=1 - (self.train_split + self.validation_split),
        random_state=self.seed,
    )

    training = DataSplit(
        image_data=images_train,
        numerical=num_train,
        targets=targets_train
    )
    validation = DataSplit(
        image_data=images_val,
        numerical=num_val,
        targets=targets_val
    )

    return TrainValData(
        train=training,
        val=validation
    )


class DataLoader(BaseDataLoader):
  def _process_data(self, files_chunk: List[str]) -> TrainValData:
    """
    Process data by converting images to grayscale and extracting pixels

    Args:
      files: List of filenames in the data directory

    Returns:
      Processed and split data for training
    """
    start_time = ti.time()
    print(f'Processing {len(files_chunk)} images ...')

    # Pre-extract metadata and filter valid bursts
    valid_bursts: List[Tuple[List[str], List[FileMetadata]]] = []
    for i in range(0, len(files_chunk), self.frames):
      burst_files = files_chunk[i:i + self.frames]
      files_metadata = [self._extract_metadata_from_filename(f) for f in burst_files]
      if self._validate(files_metadata):
        valid_bursts.append((burst_files, files_metadata))

    num_bursts = len(valid_bursts)

    # Preallocate arrays for better memory efficiency
    images_array: NDArray[np.floating] = np.empty(
        (num_bursts, self.frames, self.image_height, self.image_width, self.channels),
        dtype=np.float32
    )
    time_array: NDArray[np.floating] = np.empty(num_bursts, dtype=np.float32)
    positions_array: NDArray[np.floating] = np.empty((num_bursts, 3), dtype=np.float32)
    targets_array: NDArray[np.floating] = np.empty((num_bursts, 4), dtype=np.float32)

    # Load images sequentially
    for idx, (burst_files, metadata) in enumerate(valid_bursts):
      images_array[idx] = np.stack([self._get_pixels(f) for f in burst_files])
      time_array[idx] = metadata[-1].elapsed_time
      positions_array[idx] = metadata[-1].sat_position
      targets_array[idx] = metadata[-1].sat_rotation

    # Combine time and positions, then normalize together
    numerical_data: NDArray[np.floating] = np.column_stack([time_array, positions_array])
    scaler: StandardScaler = StandardScaler()
    numerical_data_norm: NDArray[np.floating] = scaler.fit_transform(numerical_data)

    end_time = ti.time()
    execution_time = end_time - start_time
    print(f'Processing time: {execution_time:.4f} seconds')

    return self._split_data(images_array, numerical_data_norm, targets_array)
