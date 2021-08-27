import argparse
import cv2 as cv
import os
import numpy as np
import progressbar

from utility import ensure_dir, load_filenames

def stitch_images(in_dir, out_dir, debug):
    ensure_dir(out_dir)
    if debug:
        ensure_dir(out_dir + "/debug")
    filenames = load_filenames(in_dir)
    outimg = cv.imread(filenames[0])

    for idx, filename in progressbar.progressbar(enumerate(filenames[1:]), max_value=len(filenames) -  1):
        img = cv.imread(filename)
        outimg, img = fill_holes(outimg, img)

        if debug:
            cv.imwrite(out_dir + "/debug/{}_first.png".format(idx), outimg)
            cv.imwrite(out_dir + "/debug/{}_second.png".format(idx), img)

        i = float(idx + 1)
        weight1 = i / (i + 1)
        weight2 = 1 / (i + 1)
        outimg = add_images(outimg, weight1, img, weight2)

        if debug:
            cv.imwrite(out_dir + "/debug/{}_combined.png".format(idx), outimg)

    cv.imwrite(os.path.join(out_dir, "stitched.png"), outimg)

def add_images(img1, weight1, img2, weight2):
    gray1 = cv.cvtColor(img1, cv.COLOR_BGR2GRAY)
    gray2 = cv.cvtColor(img2, cv.COLOR_BGR2GRAY)

    _, thresh1 = cv.threshold(gray1,1,255,cv.THRESH_BINARY)
    _, thresh2 = cv.threshold(gray2,1,255,cv.THRESH_BINARY)

    mask = cv.bitwise_and(thresh1, thresh2)

    masked1 = cv.bitwise_and(img1,img1,mask = mask)
    masked2 = cv.bitwise_and(img2,img2,mask = mask)

    mask_inv1 = cv.bitwise_xor(mask, thresh1)
    mask_inv2 = cv.bitwise_xor(mask, thresh2)

    masked_inv1 = cv.bitwise_and(img1,img1,mask = mask_inv1)
    masked_inv2 = cv.bitwise_and(img2,img2,mask = mask_inv2)

    weighted = cv.addWeighted(masked1, weight1, masked2, weight2, 0)

    tmp = cv.bitwise_or(masked_inv1, weighted)

    return cv.bitwise_or(masked_inv2, tmp)

def scale_contour(contour, scale):
    """Shrinks or grows a contour by the given factor (float). 
    Returns the resized contour"""
    moments = cv.moments(contour)
    midX = int(round(moments["m10"] / moments["m00"]))
    midY = int(round(moments["m01"] / moments["m00"]))
    mid = np.array([midX, midY])
    contour = contour - mid
    contour = (contour * scale).astype(np.int32)
    contour = contour + mid
    return contour

def shrink_mask(img, scale):
    contours, _ = cv.findContours(img.copy(), cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    contour_img = np.zeros_like(img)
    cv.drawContours(contour_img, contours, 0, 255, thickness=cv.FILLED)

    contour_scaled = scale_contour(contours[0], scale)
    contour_img_scaled = np.zeros_like(img)
    cv.drawContours(contour_img_scaled, [contour_scaled], 0, 255, thickness=cv.FILLED)

    borders = contour_img - contour_img_scaled

    return img - borders

def fill_holes(img1, img2):
    # Add pixels that are only contained in image 1 to image 2 and vice versa
    gray1 = cv.cvtColor(img1, cv.COLOR_BGR2GRAY)
    gray2 = cv.cvtColor(img2, cv.COLOR_BGR2GRAY)

    _, thresh1 = cv.threshold(gray1,20,255,cv.THRESH_BINARY)
    _, thresh2 = cv.threshold(gray2,20,255,cv.THRESH_BINARY)

    thresh1_shrunk = shrink_mask(thresh1, 0.98)
    _, thresh1_shrunk = cv.threshold(thresh1_shrunk,200,255,cv.THRESH_BINARY)
    thresh2_shrunk = shrink_mask(thresh2, 0.98)
    _, thresh2_shrunk = cv.threshold(thresh2_shrunk,200,255,cv.THRESH_BINARY)

    mask1 = thresh1 - thresh2_shrunk
    mask2 = thresh2 - thresh1_shrunk
    _, mask1 = cv.threshold(mask1,200,255,cv.THRESH_BINARY)
    _, mask2 = cv.threshold(mask2,200,255,cv.THRESH_BINARY)

    mask1_inv = cv.bitwise_not(mask1)
    mask2_inv = cv.bitwise_not(mask2)

    img1_cropped = cv.bitwise_and(img1,img1,mask = mask2_inv)
    img2_cropped = cv.bitwise_and(img2,img2,mask = mask1_inv)

    masked1 = cv.bitwise_and(img1,img1,mask = mask1)
    masked2 = cv.bitwise_and(img2,img2,mask = mask2)

    res1 = cv.bitwise_or(img1_cropped,masked2)
    res2 = cv.bitwise_or(img2_cropped,masked1)

    return res1, res2

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Load stabilized images and generate single image of entire scene.")
    parser.add_argument("-i", "--input-dir", type=str, required=True, help="Image directory.")
    parser.add_argument("-o", "--output-dir", type=str, default="/project/output", help="Output directory.")
    parser.add_argument("-d", "--debug", action="store_true", help="Whether to produce debug images in output_dir/debug.")

    args = parser.parse_args()

    stitch_images(args.input_dir, args.output_dir, args.debug)
