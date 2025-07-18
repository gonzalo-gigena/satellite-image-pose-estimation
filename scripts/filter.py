import argparse
import os

import numpy as np
from PIL import Image


def is_image_empty(image_path: str, threshold: float = 0.95, darkness_threshold: int = 10) -> bool:
  """
  Check if an image is mostly black (empty)

  Args:
    image_path (str): Path to the image file
    threshold (float): Threshold for determining if image is empty
    darkness_threshold (int): Pixel value below which a pixel is considered black

  Returns:
    bool: True if image is mostly black, False otherwise
  """
  try:
    # Open the image
    img: Image.Image = Image.open(image_path)

    # Convert image to grayscale
    img_gray: Image.Image = img.convert("L")

    # Convert to numpy array
    img_array: np.ndarray = np.array(img_gray)

    # Calculate the percentage of dark pixels
    dark_pixels: int = int(np.sum(img_array < darkness_threshold))
    total_pixels: int = int(img_array.size)

    dark_ratio: float = dark_pixels / total_pixels

    return dark_ratio > threshold

  except Exception as e:
    print(f"Error processing {image_path}: {str(e)}")
    return False


def delete_empty_images(folder_path: str, threshold: float, delete: bool, frames: int) -> None:
  """
  Check all images in a folder and delete empty (mostly black) images

  Args:
    folder_path (str): Path to the folder containing images
    threshold (float): Threshold for determining if image is empty
    delete (bool): Whether to delete the images
    frames (int): Number of frames per burst
  """
  # Supported image extensions
  valid_extensions: tuple[str, ...] = (".jpg",)

  # Check if folder exists
  if not os.path.exists(folder_path):
    print(f"Error: Folder {folder_path} does not exist!")
    return

  files: list[str] = [f for f in os.listdir(folder_path) if f.startswith("cubesat")]
  files.sort()  # The order of files is important

  # Iterate through all files in the folder
  for i in range(0, len(files), frames):
    are_empty: bool = True
    images: list[str] = []
    for j in range(frames):
      if files[i + j].lower().endswith(valid_extensions):
        file_path: str = os.path.join(folder_path, files[i + j])
        are_empty = are_empty and is_image_empty(file_path, threshold)
        images.append(file_path)

    if delete and are_empty:
      for j in range(frames):
        try:
          if delete:
            os.remove(images[j])
        except Exception as e:
          print(f"Error deleting {images[j]}: {str(e)}")


def parse_arguments() -> argparse.Namespace:
  """
  Parse command line arguments

  Returns:
      argparse.Namespace: Parsed command line arguments
  """
  parser: argparse.ArgumentParser = argparse.ArgumentParser(
      description="Delete empty (mostly black) images from a folder"
  )

  parser.add_argument("--path", type=str, required=True, help="Path to the folder containing images")

  parser.add_argument(
      "--threshold", type=float, default=0.95, help="Threshold for determining if an image is empty (0.0 to 1.0)"
  )

  parser.add_argument("--frames", type=int, default=3, help="Number of frames per burst")

  parser.add_argument("--delete", action="store_true", default=True, help="Delete images below the threshold")

  args: argparse.Namespace = parser.parse_args()

  # Validate threshold
  if not 0 <= args.threshold <= 1:
    parser.error("Threshold must be between 0.0 and 1.0")

  return args


if __name__ == "__main__":
  args = parse_arguments()
  delete_empty_images(args.path, args.threshold, args.delete, args.frames)
