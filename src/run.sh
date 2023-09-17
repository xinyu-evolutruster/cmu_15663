#!/bin/bash

# Find bayer pattern
# python main.py -f './data/campus.tiff' -m find_bayer_pattern

# Manual white balancing
# python main.py -f './data/campus.tiff' -m manual_white_balancing

# Show every step of the image processing pipeline
python main.py -f './data/campus.tiff' -m processing -wb true -wm mean -d true -c true -b true -pi 0.3 -g true