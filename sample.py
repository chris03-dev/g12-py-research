# If OpenCV is not installed, try 'pip install opencv-python'
# USAGE: sample.py <filename>

import cv2
import sys

# Open image
image = cv2.imread(sys.argv[1].encode('unicode-escape').decode().replace('\\\\', '\\'))

b = image.copy()
# set green and red channels to 0
b[:, :, 1] = 0
b[:, :, 2] = 0

g = image.copy()
# set blue and red channels to 0
g[:, :, 0] = 0
g[:, :, 2] = 0

r = image.copy()
# set blue and green channels to 0
r[:, :, 0] = 0
r[:, :, 1] = 0

# RGB - Blue
cv2.imshow('B-RGB', b)
cv2.waitKey(0)

# RGB - Green
cv2.imshow('G-RGB', g)
cv2.waitKey(0)

# RGB - Red
cv2.imshow('R-RGB', r)
cv2.waitKey(0)
cv2.destroyAllWindows()