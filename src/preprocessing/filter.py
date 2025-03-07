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

def delete_empty_images(folder_path, threshold, delete):
  """
  Check all images in a folder and delete empty (mostly black) images
  
  Args:
    folder_path (str): Path to the folder containing images
    threshold (float): Threshold for determining if image is empty
  """
  # Supported image extensions
  valid_extensions = ('.png', '.jpg', '.jpeg', '.tiff', '.bmp')
  
  # Check if folder exists
  if not os.path.exists(folder_path):
    print(f"Error: Folder {folder_path} does not exist!")
    return
  
  # Iterate through all files in the folder
  for filename in os.listdir(folder_path):
    if filename.lower().endswith(valid_extensions):
      file_path = os.path.join(folder_path, filename)
      
      if is_image_empty(file_path, threshold):
        try:
          if delete:
            os.remove(file_path)
        except Exception as e:
          print(f"Error deleting {filename}: {str(e)}")

def parse_arguments():
  """
  Parse command line arguments
  
  Returns:
    argparse.Namespace: Parsed command line arguments
  """
  parser = argparse.ArgumentParser(description='Delete empty (mostly black) images from a folder')
  
  parser.add_argument('--path', 
                    type=str,
                    required=True,
                    help='Path to the folder containing images')
  
  parser.add_argument('--threshold',
                    type=float,
                    default=0.95,
                    help='Threshold for determining if an image is empty (0.0 to 1.0)')
  
  parser.add_argument('--delete',
                    action='store_true',
                    help='Delete images below the threshold')
  
  args = parser.parse_args()
  
  # Validate threshold
  if not 0 <= args.threshold <= 1:
    parser.error("Threshold must be between 0.0 and 1.0")
  
  return args

if __name__ == "__main__":
  args = parse_arguments()
  
  delete_empty_images(args.path, args.threshold, args.delete)