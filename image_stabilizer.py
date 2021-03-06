import os
import time
import argparse
from pathlib import Path
import numpy as np
import cv2 as cv

from utility import ensure_dir, load_filenames

"""
Script to stabilize images using homography estimation.
"""

MIN_MATCH_COUNT = 10

def compute_homography_list(filenames, debug_dir = ""):
  sift = cv.SIFT_create()
  FLANN_INDEX_KDTREE = 1
  index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
  search_params = dict(checks = 50)
  flann = cv.FlannBasedMatcher(index_params, search_params)

  img1 = cv.imread(filenames[0])
  kp1, des1 = sift.detectAndCompute(img1,None)
  shape = (img1.shape[0], img1.shape[1])

  homography_list = []

  for i in range(1, len(filenames)):
    start = time.time()

    img2 = cv.imread(filenames[i])
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
      H, _ = cv.findHomography(pts2, pts1, method=cv.RANSAC)
      homography_list.append(H)
      end = time.time()
      print("Computed homography {}/{} in {:.3f} s".format(i, len(filenames)-1, end - start))
    else:
      print( "Not enough matches are found for image {} - {}/{}".format(filenames[i], len(good), MIN_MATCH_COUNT) )

    kp1 = kp2
    des1 = des2

  return homography_list, shape

def chain_homography(homography_list):
  # Chain homography matrices
  for i in range(1, len(homography_list)):
    homography_list[i] = np.matmul(homography_list[i-1], homography_list[i])

  return homography_list

def find_image_shape(homography_list, shape):
  # Find the corners after the transform has been applied
  height, width = shape
  corners_raw = np.array([
    [0, 0],
    [0, height - 1],
    [width - 1, height - 1],
    [width - 1, 0]
  ])
  u0_min = 0
  bwidth_max = width
  v0_min = 0
  bheight_max = height

  # Find maximum translation
  corners_transformed = []
  for i in range(0, len(homography_list)):
    corners = cv.perspectiveTransform(np.float32([corners_raw]), homography_list[i])[0]
    corners_transformed.append(corners)
    u0, v0, _, _ = cv.boundingRect(corners)
    u0_min = min(u0_min, u0)
    v0_min = min(v0_min, v0)

  translation = (u0_min, v0_min)

  # Find maximum bounding box after translation
  for corners in corners_transformed:
    for idx in range(0, len(corners)):
      corners[idx, 0] = corners[idx, 0] - u0_min
      corners[idx, 1] = corners[idx, 1] - v0_min
    corners = np.vstack((corners, np.zeros((1, 2))))

    _, _, bwidth, bheight = cv.boundingRect(np.float32(corners))
    bwidth_max = max(bwidth_max, bwidth)
    bheight_max = max(bheight_max, bheight)

  width = bwidth_max
  height = bheight_max

  # Images should have dimensions divisible by 2 for most video codecs
  if width % 2 == 1:
    width = width + 1
  if height % 2 == 1:
    height = height + 1
  new_shape = (height, width)

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

def warp_perspective(filenames, homography_list, shape, out_dir, debug_dir = ""):
  for idx, (f, h) in enumerate(zip(filenames, homography_list)):
    start = time.time()
    img = cv.imread(f)
    corrected = cv.warpPerspective(img, h, (shape[1], shape[0]))
    filename = Path(f)
    filename = filename.stem
    cv.imwrite(os.path.join(out_dir, filename + ".png"), corrected)

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

def debug_draw_features(img, kp, i, debug_dir):
  debug_img = img
  for idx, p in enumerate(kp):
    p = (int(p[0][0]), int(p[0][1]))
    cv.circle(debug_img, p, 5, (255, 0, 0), -1)

  cv.imwrite(os.path.join(debug_dir, str(i) + "_features.png"), debug_img)

def debug_draw_before_after(ref_before, ref_after, before, after, i, debug_dir):
  combined_before = cv.addWeighted(ref_before,0.4,before,0.2,0)
  combined_after = cv.addWeighted(ref_after,0.4,after,0.2,0)

  cv.imwrite(os.path.join(debug_dir, str(i) + "_before.png"), combined_before)
  cv.imwrite(os.path.join(debug_dir, str(i) + "_after.png"), combined_after)

def debug_draw_matches(img1, img2, pts1, pts2, i, debug_dir):
  debug_img = cv.hconcat([img1, img2])
  width = img1.shape[1]
  for idx, (pt1, pt2) in enumerate(zip(pts1, pts2)):
    p1 = (int(pt1[0][0]), int(pt1[0][1]))
    p2 = (int(pt2[0][0])+width, int(pt2[0][1]))

    # To make lines more distinguishable, switch up the colors a little bit
    base_color = idx % 3
    color = idx * 255/(len(pts1))
    if base_color == 0:
      cv.line(debug_img,p1, p2,(color,0,0),2)
    elif base_color == 1:
      cv.line(debug_img,p1, p2,(0,color,0),2)
    elif base_color == 2:
      cv.line(debug_img,p1, p2,(0,0,color),2)

  cv.imwrite(os.path.join(debug_dir, str(i) + "_matches.png"), debug_img)

def main(in_dir, out_dir, debug):
  ensure_dir(out_dir)
  filenames = load_filenames(in_dir)
  debug_dir = ""
  if debug:
    debug_dir = os.path.join(out_dir, "debug")
    ensure_dir(debug_dir)

  homography_list, shape = compute_homography_list(filenames, debug_dir)
  homography_chained = chain_homography(homography_list)
  new_shape, translation = find_image_shape(homography_chained, shape)
  homography_chained = adjust_translation(homography_chained, translation)
  warp_perspective(filenames, homography_chained, new_shape, out_dir, debug_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Load folder of images and stabilize camera movement.")
    parser.add_argument("-i", "--input-dir", type=str, required=True, help="Image directory.")
    parser.add_argument("-o", "--output-dir", type=str, default="/project/output", help="Output directory.")
    parser.add_argument("-d", "--debug", action="store_true", help="Whether to produce debug images in output_dir/debug.")

    args = parser.parse_args()

    main(args.input_dir, args.output_dir, args.debug)
