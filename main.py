# -*- coding: utf-8 -*-

import cv2 as cv
import numpy as np
import os
import roi


img = cv.imread(os.path.join('dataset', 'image17.jpg'))


# apply CLAHE only to the luminance channel in the LAB color space
# this way we increase contrast without impacting colors so much

#  Converting image to LAB Color model
lab= cv.cvtColor(img, cv.COLOR_BGR2LAB)

# Splitting the LAB image to different channels
l, a, b = cv.split(lab)

# Applying CLAHE to L-channel
clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
cl = clahe.apply(l)

# Merge the CLAHE enhanced L-channel with the a and b channel
limg = cv.merge((cl,a,b))

# Converting image from LAB Color model to RGB model
lab_clahe = cv.cvtColor(limg, cv.COLOR_LAB2BGR)


roi.findRoi(lab_clahe, img)