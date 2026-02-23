import os
import time as ti
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
from numpy.typing import NDArray
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from config.model_config import ModelConfig


@dataclass
class FileMetadata:
  sat_name: str
  sat_index: str
  elapsed_time: int
  sat_position: NDArray[np.floating]   # shape (3,)
  sat_rotation: NDArray[np.floating]   # shape (4,)


@dataclass
class DataSplit:
  images: List[List[str]]
  numerical: NDArray[np.floating]
  targets: NDArray[np.floating]
  indices: NDArray[np.integer]


@dataclass
class TrainValTestData:
  train: DataSplit
  test: DataSplit
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
    self.test_split = config.test_split
    self.validation_split = config.validation_split
    self.seed = config.seed
    self.channels = config.channels
    self.frames = config.frames

  def load_files(self) -> None:
    files: List[str] = [f for f in os.listdir(self.data_path) if f.startswith('cubesat')]
    files.sort()  # The order of files is important for loading the images
    return files

  def load_data(self, files_chunk) -> TrainValTestData:
    """
    Load data based on file extension and type.

    Returns:
      Dictionary containing train and validation data splits
    """
    return self._process_data(files_chunk)

  def _process_data(self, files_chunk: List[str]) -> TrainValTestData:
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

    if len(data) != self.frames:
      return False

    first_index = data[0].sat_index

    return all(d.sat_index == first_index for d in data)

  def _split_data(
      self,
      burst_paths: List[List[str]],
      numerical_data: NDArray[np.float32],
      targets: NDArray[np.float32]
  ) -> TrainValTestData:

    indices = np.arange(len(burst_paths))

    train_idx, temp_idx = train_test_split(
        indices,
        test_size=1 - self.train_split,
        random_state=self.seed,
        shuffle=True
    )

    val_ratio = self.validation_split / (self.validation_split + self.test_split)

    val_idx, test_idx = train_test_split(
        temp_idx,
        test_size=1 - val_ratio,
        random_state=self.seed,
        shuffle=True
    )

    training = DataSplit(
        images=burst_paths,
        numerical=numerical_data,
        targets=targets,
        indices=train_idx
    )

    validation = DataSplit(
        images=burst_paths,
        numerical=numerical_data,
        targets=targets,
        indices=val_idx
    )

    testing = DataSplit(
        images=burst_paths,
        numerical=numerical_data,
        targets=targets,
        indices=test_idx
    )

    return TrainValTestData(
        train=training,
        val=validation,
        test=testing
    )


class DataLoader(BaseDataLoader):
  def _process_data(self, files_chunk: List[str]) -> TrainValTestData:
    """
    Process data by converting images to grayscale and extracting pixels

    Args:
      files: List of filenames in the data directory

    Returns:
      Processed and split data for training
    """
    start_time = ti.time()
    print(f'Processing {len(files_chunk)} images ...')

    # Pre-extract metadata and filter valid bursts (store only last frame's metadata)
    valid_bursts: List[Tuple[List[str], FileMetadata]] = []
    for i in range(0, len(files_chunk), self.frames):
      burst_files = files_chunk[i:i + self.frames]
      files_metadata = [self._extract_metadata_from_filename(f) for f in burst_files]

      if self._validate(files_metadata):
        valid_bursts.append((burst_files, files_metadata[-1]))

    burst_paths: List[List[str]] = []
    time_array = []
    positions_array = []
    targets_array = []

    # Load images sequentially (direct assignment avoids np.stack intermediate copies)
    for burst_files, last_metadata in valid_bursts:
      burst_paths.append([os.path.join(self.data_path, f) for f in burst_files])
      time_array.append(last_metadata.elapsed_time)
      positions_array.append(last_metadata.sat_position)
      targets_array.append(last_metadata.sat_rotation)

    numerical_data = np.column_stack([time_array, positions_array]).astype(np.float32)
    targets_array = np.array(targets_array, dtype=np.float32)

    scaler = StandardScaler()
    numerical_data = scaler.fit_transform(numerical_data).astype(np.float32)

    end_time = ti.time()
    execution_time = end_time - start_time
    print(f'Processing time: {execution_time:.4f} seconds')

    return self._split_data(burst_paths, numerical_data, targets_array)
