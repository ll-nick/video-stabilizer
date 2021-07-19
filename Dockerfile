FROM ubuntu:20.04

# To suppress user input during tzdata installation
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && \
    apt-get install --no-install-recommends -y python3-pip ffmpeg libsm6 libxext6 && \
    pip install setuptools && \
    pip install --upgrade setuptools pip && \
    pip install opencv-python

COPY image_stabilizer.py ./

ENTRYPOINT ["python3", "image_stabilizer.py", "--input-dir", "/project/input", "--output-dir", "/project/output"]