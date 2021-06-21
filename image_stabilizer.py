# Import numpy and OpenCV
from pathlib import Path
import os
import time
import glob
import argparse
import numpy as np
import cv2

"""
Script to stabilize images using homography estimation.
"""

MIN_MATCH_COUNT = 10

def compute_homography_list(filenames, debug_dir = ""):
  sift = cv2.SIFT_create()
  FLANN_INDEX_KDTREE = 1
  index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
  search_params = dict(checks = 50)
  flann = cv2.FlannBasedMatcher(index_params, search_params)

  img1 = cv2.imread(filenames[0])
  kp1, des1 = sift.detectAndCompute(img1,None)
  shape = (img1.shape[0], img1.shape[1])

  homography_list = []

  for i in range(1, len(filenames)):
    start = time.time()

    img2 = cv2.imread(filenames[i])
    kp2, des2 = sift.detectAndCompute(img2,None)

    matches = flann.knnMatch(des1,des2,k=2)
    good = filterMatches(matches)

    if len(good)>MIN_MATCH_COUNT:
      pts1 = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
      pts2 = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)
      if debug_dir:
        filename = Path(filenames[i])
        filename = filename.stem
        debug_draw_features(img1, pts2, filename, debug_dir)
        debug_draw_matches(img1, img2, pts1, pts2, filename, debug_dir)

      #Find homography matrix
      H, _ = cv2.findHomography(pts2, pts1, method=cv2.RANSAC)
      homography_list.append(H)
      end = time.time()
      print("Computed homography {}/{} in {:.3f} s".format(i, len(filenames)-1, end - start))
    else:
      print( "Not enough matches are found for image {} - {}/{}".format(filenames[i], len(good), MIN_MATCH_COUNT) )
      matchesMask = None

    kp1 = kp2
    des1 = des2

  return homography_list, shape

def chain_homography(homography_list, shape):
  # Chain homography matrices
  for i in range(1, len(homography_list)):
    homography_list[i] = np.matmul(homography_list[i-1], homography_list[i])

  return homography_list

def find_image_shape(homography_list, shape):
  # Find the corners after the transform has been applied
  height, width = shape
  corners = np.array([
    [0, 0],
    [0, height - 1],
    [width - 1, height - 1],
    [width - 1, 0]
  ])
  u0_min = 0
  bwidth_max = width
  v0_min = 0
  bheight_max = height

  # Find maximal bounding boxes
  for i in range(0, len(homography_list)):
    corners = cv2.perspectiveTransform(np.float32([corners]), homography_list[i])[0]
    u0, v0, bwidth, bheight = cv2.boundingRect(corners)
    u0_min = min(u0_min, u0)
    bwidth_max = max(bwidth_max, bwidth)
    v0_min = min(v0_min, v0)
    bheight_max = max(bheight_max, bheight)

  width = bwidth_max
  height = bheight_max

  # Images should have dimensions divisible by 2 for most video codecs
  if width % 2 == 1:
    width = width + 1
  if height % 2 == 1:
    height = height + 1
  new_shape = (height, width)
  translation = (u0_min, v0_min)

  return new_shape, translation

def adjust_translation(homography_list, translation):
  u0, v0 = translation

  mat_translation = np.array([
    [ 1, 0, -u0 ],
    [ 0, 1, -v0 ],
    [ 0, 0,   1 ]
  ], dtype = np.float32)

  # Add translation to all homographies
  for i in range(0, len(homography_list)):
    homography_list[i] = np.matmul(mat_translation, homography_list[i])
  homography_list.insert(0, mat_translation)

  return homography_list

def warp_perspective(filenames, homography_chained, shape, out_dir, debug_dir = ""):
  for idx, (f, h) in enumerate(zip(filenames, homography_chained)):
    start = time.time()
    img = cv2.imread(f)
    corrected = cv2.warpPerspective(img, h, (shape[1], shape[0]))
    filename = Path(f)
    filename = filename.stem
    cv2.imwrite(os.path.join(out_dir, filename + "_corrected.png"), corrected)

    end = time.time()
    print("Warped image {}/{} in {:.3f} s".format(idx+1, len(filenames), end - start))

    if idx == 0 and debug_dir:
      ref_before = img
      ref_after = corrected

    if debug_dir:
      debug_draw_before_after(ref_before, ref_after, img, corrected, filename, debug_dir)

def filterMatches(matches):
  # store all the good matches as per Lowe's ratio test.
  good = []
  for m,n in matches:
      if m.distance < 0.7*n.distance:
          good.append(m)
  return good

def load_filenames(dir):
  filenames = glob.glob("{}/*.png".format(dir)) 
  if len(filenames) == 0:
    raise RuntimeError("No images found in specified directory.")
  filenames.sort()

  return filenames

def ensure_dir(d):
  if not os.path.isdir(d):
      Path(d).mkdir(parents=True)

def debug_draw_features(img, kp, i, debug_dir):
  debug_img = img
  for idx, p in enumerate(kp):
    p = (int(p[0][0]), int(p[0][1]))
    cv2.circle(debug_img, p, 5, (255, 0, 0), -1)

  cv2.imwrite(os.path.join(debug_dir, str(i) + "_features.png"), debug_img)

def debug_draw_before_after(ref_before, ref_after, before, after, i, debug_dir):
  combined_before = cv2.addWeighted(ref_before,0.4,before,0.2,0)
  combined_after = cv2.addWeighted(ref_after,0.4,after,0.2,0)

  cv2.imwrite(os.path.join(debug_dir, str(i) + "_before.png"), combined_before)
  cv2.imwrite(os.path.join(debug_dir, str(i) + "_after.png"), combined_after)

def debug_draw_matches(img1, img2, pts1, pts2, i, debug_dir):
  debug_img = cv2.hconcat([img1, img2])
  width = img1.shape[1]
  for idx, (pt1, pt2) in enumerate(zip(pts1, pts2)):
    p1 = (int(pt1[0][0]), int(pt1[0][1]))
    p2 = (int(pt2[0][0])+width, int(pt2[0][1]))

    # To make lines more distinguishable, switch up the colors a little bit
    base_color = idx % 3
    color = idx * 255/(len(pts1))
    if base_color == 0:
      cv2.line(debug_img,p1, p2,(color,0,0),2)
    elif base_color == 1:
      cv2.line(debug_img,p1, p2,(0,color,0),2)
    elif base_color == 2:
      cv2.line(debug_img,p1, p2,(0,0,color),2)

  cv2.imwrite(os.path.join(debug_dir, str(i) + "_matches.png"), debug_img)

def main(in_dir, out_dir, debug):
  ensure_dir(out_dir)
  filenames = load_filenames(in_dir)
  debug_dir = ""
  if debug:
    debug_dir = os.path.join(out_dir, "debug")
    ensure_dir(debug_dir)

  homography_list, shape = compute_homography_list(filenames, debug_dir)
  homography_chained = chain_homography(homography_list, shape)
  new_shape, translation = find_image_shape(homography_list, shape)
  homography_list = adjust_translation(homography_list, translation)
  warp_perspective(filenames, homography_chained, new_shape, out_dir, debug_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Load folder of images and stabilize camera movement.")
    parser.add_argument("-i", "--input-dir", type=str, required=True, help="Image directory.")
    parser.add_argument("-o", "--output-dir", type=str, default="/project/output", help="Output directory.")
    parser.add_argument("-d", "--debug", action="store_true", help="Whether to produce debug images in output_dir/debug.")

    args = parser.parse_args()

    main(args.input_dir, args.output_dir, args.debug)
