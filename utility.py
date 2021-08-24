import os
import glob
import re

'''
Checks if given path is a file or directory based on if last element of path contains a '.'
If the directory (or the directory of the given file) does not exist, it will be created.
'''
def ensure_dir(d):
    if len(os.path.split(d)[-1].split('.')) > 1:
        d = os.path.dirname(d)
    if not os.path.exists(d):
        os.makedirs(d)

def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    '''
    alist.sort(key=natural_keys) sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html
    (See Toothy's implementation in the comments)
    '''
    return [ atoi(c) for c in re.split(r'(\d+)', text) ]

def load_filenames(dir, ext = "png"):
  filenames = glob.glob("{}/*.{}".format(dir, ext)) 
  if len(filenames) == 0:
    raise RuntimeError("No images found in specified directory.")
  filenames.sort(key=natural_keys)

  return filenames