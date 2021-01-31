# -*- coding: utf-8 -*-

import cv2 as cv
import numpy as np
import os
import roi
import detect as det


img = cv.imread(os.path.join('dataset', 'image34.jpg'))

# pre-process the image to increase contrast and reduce noise
pre = roi.preprocess(img)

# find contours of candidate signs
contours = roi.findRoi(pre)

if len(contours) == 0:
    print('No sign detected')
else:
    # visualize contours
    vis = img.copy()
    cv.drawContours(vis, contours, -1, (0,255,0), 2)
    cv.imshow('IMG', vis)
    cv.waitKey(0)
    cv.destroyAllWindows()

    # perspective correction
    signs = roi.correctPerspective(contours, img)

    print('Detecting signs')

    # identify signs keeping the best match
    for s in signs:
        found = det.detectSign(s)
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
            detected = 'No sign detected'
        else:
            print("Found", detected)
        
        cv.imshow(detected, s)

    print('Done')
    cv.waitKey(0)
    cv.destroyAllWindows()

