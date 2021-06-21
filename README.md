# video-stabilizer
Small project used to stabilize images/videos using homography computation.

Currently works with already extracted png-images. So use a tool such as ffmpeg to extract the images from a video you want stabilized.
Then start the container using
```
docker run -t --rm -v <path/to/input/data>:/project/input -v <path/to/output/directory>:/project/output nickll/video-stabilizer
```

The script will store the stabilized images into the specified directory.
Again, if you'd like to have a video from that, use ffmpeg.

The script will _not_ crop the images but rather find the required size to pad the transformed images with the respective number of black pixels.
Therefore, if there is a lot of movement, the images will be rather large.

Since the script is based an homographic transformations, it assumes that all pixels lie on a flat plane.
For large 3-dimensional movements the results are likely going to be bad.