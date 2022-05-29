'''testing script to check if a given image is day image or night image
the selection of threshold is explain in optimal_threshold_selection.py
to use this script run python3 test_image.py path_to_image'''

from PIL import Image #reading images
import os #iterating over folders
import numpy as np #performing calculations
import argparse

#parsing input arguments
parser = argparse.ArgumentParser(description='Day or Night Image')
parser.add_argument('image', metavar='image',
                    help='path to image')
args = parser.parse_args()

#performing inference steps
img = Image.open(args.image)
img = img.resize((800,800))
img = img.convert('HSV')
img = np.asarray(img)
v = np.sum(img[:, :, 2])
v = v/640000

#checking if greater than or less than threshold
threshold = 108
if v>threshold:
    print("Day")
else:
    print("Night")