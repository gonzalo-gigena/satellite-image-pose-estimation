import numpy as np

""" Camera Module v2 for raspberry pi
  'Net price': '$25',
  'Size': 'Around 25 × 24 × 9 mm',
  'Weight': '3g',
  'Still resolution': '8 megapixels',
  'Video modes': '1080p47, 1640 × 1232p41 and 640 × 480p206',
  'Sensor': 'Sony IMX219',
  'Sensor resolution': '3280 × 2464 pixels',
  'Sensor image area': '3.68 × 2.76 mm (4.6 mm diagonal)',
  'Pixel size': '1.12 µm × 1.12 µm',
  'Optical size': '1/4"',
  'Focus': 'Adjustable',
  'Depth of field': 'Approx 10 cm to ∞',
  'Focal length': '3.04 mm',
  'Horizontal Field of View (FoV)': '62.2 degrees',
  'Vertical Field of View (FoV)': '48.8 degrees',
  'Focal ratio (F-Stop)': 'F2.0',
  'Maximum exposure time (seconds)': '11.76',
  'Lens Mount': 'N/A',
  'NoIR version available?': 'Yes'
"""

def get_camera_matrix():
  # Physical sensor specifications
  sensor_width_mm = 3.68
  sensor_height_mm = 2.76
  focal_length_mm = 3.04

  # Actual capture resolution
  capture_width = 102
  capture_height = 102

  # Calculate focal length in pixels for the 102x102 capture
  fx = (focal_length_mm * capture_width) / sensor_width_mm
  fy = (focal_length_mm * capture_height) / sensor_height_mm

  # Principal point (center of the 102x102 image)
  cx = capture_width / 2  # 51
  cy = capture_height / 2 # 51

  # Create camera matrix for 102x102 capture
  camera_matrix = np.array([
      [fx,  0,  cx],
      [0,  fy,  cy],
      [0,   0,   1]
  ])
  return camera_matrix