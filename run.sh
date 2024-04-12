#!/bin/bash

# Build the Docker image
docker build -t my_jupyter_image .

# Run the Docker container
docker run -p 8000:8000 -v $(pwd):/app  my_jupyter_image
#docker run -p 8000:8000 -v $(pwd):/.  my_jupyter_i
# mage
