import argparse
import cv2 as cv
import os

from utility import ensure_dir, load_filenames

def stitch_images(in_dir, out_dir, debug):
    ensure_dir(out_dir)
    filenames = load_filenames(in_dir)
    outimg = cv.imread(filenames[0])

    for idx, filename in enumerate(filenames[1:]):
        img = cv.imread(filename)
        outimg, img = fill_holes(outimg, img)

        i = float(idx + 1)
        weight1 = i / (i + 1)
        weight2 = 1 / (i + 1)
        outimg = cv.addWeighted(outimg, weight1, img, weight2, 0.0)

    cv.imwrite(os.path.join(out_dir, "stitched.png"), outimg)

def fill_holes(img1, img2):
    gray1 = cv.cvtColor(img1, cv.COLOR_BGR2GRAY)
    gray2 = cv.cvtColor(img2, cv.COLOR_BGR2GRAY)

    _, thresh1 = cv.threshold(gray1,10,255,cv.THRESH_BINARY)
    _, thresh2 = cv.threshold(gray2,10,255,cv.THRESH_BINARY)

    mask1 = thresh1 - thresh2
    mask2 = thresh2 - thresh1

    masked1 = cv.bitwise_and(img1,img1,mask = mask1)
    masked2 = cv.bitwise_and(img2,img2,mask = mask2)

    res1 = cv.bitwise_or(img1,masked2)
    res2 = cv.bitwise_or(img2,masked1)
    return res1, res2

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Load stabilized images and generate single image of entire scene.")
    parser.add_argument("-i", "--input-dir", type=str, required=True, help="Image directory.")
    parser.add_argument("-o", "--output-dir", type=str, default="/project/output", help="Output directory.")
    parser.add_argument("-d", "--debug", action="store_true", help="Whether to produce debug images in output_dir/debug.")

    args = parser.parse_args()

    stitch_images(args.input_dir, args.output_dir, args.debug)
