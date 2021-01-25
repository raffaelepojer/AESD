# -*- coding: utf-8 -*-

import cv2 as cv
import numpy as np
import os
import roi
import detect as det
# import imutils # keeps the aspect ratio


img = cv.imread(os.path.join('dataset', 'image13.jpg'))


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

img2 = cv.imread(os.path.join('dataset', 'template', 'image5-cropped.jpg'), cv.IMREAD_COLOR)
img_gray = cv.cvtColor(img2, cv.COLOR_BGR2GRAY)
imgcopy = img2.copy()

# draw the rectangle of the template found
arr = (startXa, startYa, endXa, endYa, foundA) = det.findArrow(img2)
door = (startXd, startYd, endXd, endYd, foundD) = det.findDoor(img2)

if foundA == "NO_ARROW":
    print("No arrow found")
else:
    print("Found ", foundA, " at x: ", startXa, " y: ", startYa)
    cv.rectangle(img2, (startXa, startYa), (endXa, endYa), (0, 0, 255), 2)

if foundD == "NO_DOOR":
    print("No door found")
else:
    print("Found ", foundD, " at x: ", startXd, " y: ", startYd)
    cv.rectangle(img2, (startXd, startYd), (endXd, endYd), (255, 0, 0), 2)

cv.imshow("Image", img2)
cv.waitKey(0)

roi.findRoi(lab_clahe, img)
