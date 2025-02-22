import argparse

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