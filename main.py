# -*- coding: utf-8 -*-

import cv2 as cv
import numpy as np
import os
import roi
import detect as det
import linecache

img = cv.imread(os.path.join('dataset', 'image26.jpg'))
# label = linecache.getline('label.txt', 10).strip()

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

# blurring to reduce noise
blur = cv.GaussianBlur(lab_clahe, (5,5), 0)

contours = roi.findRoi(blur)

# vis = img.copy()
# cv.drawContours(vis, contours, -1, (0,255,0), 2)
# cv.imshow('IMG', vis)
# # cv.waitKey(0)

signs = roi.correctPerspective(contours, img)

# for s in signs:
#     cv.imshow('WARPED', s)
#     cv.waitKey(0)

for s in signs:
    found = det.detectSign(s)
    print(found)
    max = 0
    detected = ""
    direction = -1
    for x in found:
        if x != 0 and x[0] > max:
            max = x[0]
            detected = x[1]
            direction = x[2]
    if max == 0:
        print("No sign detected")
    else:
        print("Found", detected)

# img2 = cv.imread(os.path.join('dataset', 'template', 'image5-cropped.jpg'), cv.IMREAD_COLOR)
# img_gray = cv.cvtColor(img2, cv.COLOR_BGR2GRAY)
# imgcopy = img2.copy()
# get the wrapped image
# draw the rectangle of the template found
# arr = (startXa, startYa, endXa, endYa, foundA) = det.findArrow(signcopy)
# door = (startXd, startYd, endXd, endYd, foundD) = det.findDoor(signcopy)
# chair = (startXc, startYc, endXc, endYc, foundC) = det.findChair(signcopy)
# man = (startXm, startYm, endXm, endYm, foundM) = det.findMan(signcopy)

# if foundA == "NO_ARROW":
#     print("No arrow found")
# else:
#     print("Found ", foundA, " at x: ", startXa, " y: ", startYa)
#     cv.rectangle(signcopy, (startXa, startYa), (endXa, endYa), (0, 0, 255), 2)

# if foundD == "NO_DOOR":
#     print("No door found")
# else:
#     print("Found ", foundD, " at x: ", startXd, " y: ", startYd)
#     cv.rectangle(signcopy, (startXd, startYd), (endXd, endYd), (255, 0, 0), 2)

# if foundC == "NO_CHAIR":
#     print("No chair found")
# else:
#     print("Found ", foundC, " at x: ", startXc, " y: ", startYc)
#     cv.rectangle(signcopy, (startXc, startYc), (endXc, endYc), (255, 0, 255), 2)

# if foundM == "NO_MAN":
#     print("No man found")
# else:
#     print("Found ", foundM, " at x: ", startXm, " y: ", startYm)
#     cv.rectangle(signcopy, (startXm, startYm), (endXm, endYm), (0, 255, 255), 2)

# cv.imshow("Image", signcopy)
# cv.waitKey(0)

# cv.imwrite('sign0.jpg', signs[0])