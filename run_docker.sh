#!/usr/bin/env bash
docker run --gpus all --rm -it \
    --env-file ./envfile \
    -v /PATH_TO_REPO:/workspace/biomedical \
    -v /PATH_TO_IMAGES:/mnt/images \
    -p 8888:8888 -p 8887:8887 -p 8786:8786 -p 7010:7010 -p 5000:5000 \
    IMAGE_NAME