import os
from PIL import Image
import numpy as np
import argparse

def is_image_empty(image_path, threshold=0.95, darkness_threshold=10):
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
    img = Image.open(image_path)
    
    # Convert image to grayscale
    img_gray = img.convert('L')
    
    # Convert to numpy array
    img_array = np.array(img_gray)
    
    # Calculate the percentage of dark pixels
    dark_pixels = np.sum(img_array < darkness_threshold)
    total_pixels = img_array.size
    
    dark_ratio = dark_pixels / total_pixels
    
    return dark_ratio > threshold
    
  except Exception as e:
    print(f"Error processing {image_path}: {str(e)}")
    return False

def delete_empty_images(folder_path, threshold, delete, frames):
  """
  Check all images in a folder and delete empty (mostly black) images
  
  Args:
    folder_path (str): Path to the folder containing images
    threshold (float): Threshold for determining if image is empty
  """
  # Supported image extensions
  valid_extensions = ('.jpg')
  
  # Check if folder exists
  if not os.path.exists(folder_path):
    print(f"Error: Folder {folder_path} does not exist!")
    return
  
  files=  [f for f in os.listdir(folder_path) if f.startswith('cubesat')]
  files.sort() # The order of files is importat
  
  # Iterate through all files in the folder
  for i in range(0, len(files), frames):
    are_empty = True
    images = []
    for j in range(frames):
      if files[i+j].lower().endswith(valid_extensions):
        file_path = os.path.join(folder_path, files[i+j])
        are_empty = are_empty and is_image_empty(file_path, threshold)
        images.append(file_path)
        
    if delete and are_empty:
      for j in range(frames):
        try:
          if delete:
            os.remove(images[j])
        except Exception as e:
          print(f"Error deleting {images[j]}: {str(e)}")