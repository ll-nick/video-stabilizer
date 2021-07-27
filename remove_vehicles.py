import argparse
import cv2 as cv
import os
from pathlib import Path

from utility import ensure_dir, load_filenames

class Detection:
    def __init__(self, label, x, y, w, h, confidence):
        self.label = label
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        self.confidence = confidence

    def rel2abs(self, image_shape):
        height, width = image_shape[:2]
        self.x = self.x * width
        self.y = self.y * height
        self.w = self.w * width
        self.h = self.h * height

def remove_vehicles_main(img_dir, detections_path, out_dir):
    ensure_dir(out_dir)
    print("Writing output images to {}".format(out_dir))

    img_filenames = load_filenames(img_dir)
    detections_filenames = load_filenames(detections_path, "txt")

    for idx, img_filename in enumerate(img_filenames):
        print("Removing vehicles in image {}/{}".format(idx+1, len(img_filenames)))
        img = remove_vehicles(img_filename, detections_filenames)
        cv.imwrite(os.path.join(out_dir, Path(img_filename).stem + "_cleaned.png"), img)

def remove_vehicles(img_filename, detections_filenames):
    img = cv.imread(img_filename)
    stem = Path(img_filename).stem
    ret, detections = get_detections(stem, detections_filenames)

    for detection in detections:
        detection.rel2abs(img.shape)
        x = int(detection.x)
        y = int(detection.y)
        half_w = int(detection.w/2)
        half_h = int(detection.h/2)
        img[y-half_h:y+half_h, x-half_w:x+half_w, :] = 0

    return img

def get_detections(stem, detections_filenames):
    matches = [s for s in detections_filenames if stem + ".txt" in s]
    if len(matches) == 0:
        print("No matching detections found for image {}.png. Skipping image.".format(stem))
        return False, None
    if len(matches) > 1:
        print("More than one match found for image {}.png. Skipping image.".format(stem))
        return False, None

    match = matches[0]
    detections = read_detections(match)

    return True, detections

def read_detections(file):
    detections = []
    with open(file, "r") as fid:
        while True:
            line = fid.readline()
            if not line:
                break
            line = line.strip()
            if len(line) > 0 and line[0] != "#":
                elems = line.split()
                label = int(elems[0])
                x = float(elems[1])
                y = float(elems[2])
                w = float(elems[3])
                h = float(elems[4])
                confidence = float(elems[5])
                detection = Detection(
                    label=label, x=x, y=y,
                    w=w, h=h, confidence=confidence)
                detections.append(detection)
    return detections

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Load stabilized images and generate single image of entire scene.")
    parser.add_argument("-i", "--img-dir", type=str, required=True, help="Image directory.")
    parser.add_argument("-d", "--detections-dir", type=str, required=True, help="Directory containing detections.")
    parser.add_argument("-o", "--output-dir", type=str, default="/project/output", help="Output directory.")

    args = parser.parse_args()

    remove_vehicles_main(args.img_dir, args.detections_dir, args.output_dir)