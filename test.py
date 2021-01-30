# -*- coding: utf-8 -*-

import cv2 as cv
import numpy as np
import os
import roi
import detect as det
import linecache


for i in [18,20]:

    print('Image ' + str(i))

    img = cv.imread(os.path.join('dataset', 'image'+str(i)+'.jpg'))

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
    # cv.waitKey(0)
    # cv.destroyAllWindows()

    signs = roi.correctPerspective(contours, img)

    for s in signs:
        # cv.imshow('SIGN', s)
        # cv.waitKey(0)

        found = det.detectSign(s)
        # print(found)
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
        
        cv.imshow('SIGN', s)
        cv.waitKey(0)
        cv.destroyAllWindows()
