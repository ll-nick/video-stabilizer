import os
import glob
from pathlib import Path

def ensure_dir(d):
  if not os.path.isdir(d):
      Path(d).mkdir(parents=True)

def load_filenames(dir, ext = "png"):
  filenames = glob.glob("{}/*.{}".format(dir, ext)) 
  if len(filenames) == 0:
    raise RuntimeError("No images found in specified directory.")
  filenames.sort()

  return filenames