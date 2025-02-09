# Assume extract_points is defined in light_glue_model.py
try:
  from LightGlue.lightglue import LightGlue, SuperPoint, DISK, SIFT, ALIKED, DoGHardNet
  from LightGlue.lightglue.utils import load_image, rbd
except ImportError:
  raise ImportError("Failed to import 'LightGlue' from 'LightGlue.lightglue'. Please ensure the module is installed and accessible.")

# SuperPoint+LightGlue
extractor = SuperPoint(max_num_keypoints=None).eval().cuda()  # load the extractor

# or DISK+LightGlue, ALIKED+LightGlue or SIFT+LightGlue
#extractor = DISK(max_num_keypoints=2048).eval().cuda()  # load the extractor
  

# SuperPoint+LightGlue
matcher = LightGlue(features='superpoint', depth_confidence=-1, width_confidence=-1).eval().cuda()  # load the matcher

# or DISK+LightGlue, ALIKED+LightGlue or SIFT+LightGlue
#matcher = LightGlue(features='disk').eval().cuda()  # load the matcher

def extract_points( image0_path, image1_path): 
  # load each image as a torch.Tensor on GPU with shape (3,H,W), normalized in [0,1]
  image0 = load_image(image0_path, resize=None).cuda()
  image1 = load_image(image1_path, resize=None).cuda()

  # extract local features
  feats0 = extractor.extract(image0)  # auto-resize the image, disable with resize=None
  feats1 = extractor.extract(image1)

  # match the features
  matches01 = matcher({'image0': feats0, 'image1': feats1})
  feats0, feats1, matches01 = [rbd(x) for x in [feats0, feats1, matches01]]  # remove batch dimension
  matches = matches01['matches']  # indices with shape (K,2)
  points0 = feats0['keypoints'][matches[..., 0]]  # coordinates in image #0, shape (K,2)
  points1 = feats1['keypoints'][matches[..., 1]]  # coordinates in image #1, shape (K,2)
  
  return points0, points1