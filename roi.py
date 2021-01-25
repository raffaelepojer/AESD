import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt


def findRoi(img, original):
    # conversion to HSV
    hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)

    # color thresholding

    # range of green in the dataset
    # use 60, ... in lower for stricter bound
    # use 24, ... in lower for less strict bound
    lower_green = np.array([60,40,0])
    upper_green = np.array([95, 255,255])


    # Threshold the HSV image
    mask = cv.inRange(hsv, lower_green, upper_green)


    # cv.imshow('test', mask)
    # cv.waitKey(0)


    # perform closing operation to try to find more closed contours
    kernel = cv.getStructuringElement(cv.MORPH_RECT,(4,4))
    mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, kernel)


    edges = cv.Canny(mask, 50, 200)
    contours, _ = cv.findContours(edges, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)


    # approximate contours
    for i, cnt in enumerate(contours):
        epsilon = 0.1*cv.arcLength(cnt,True)
        contours[i] = cv.approxPolyDP(cnt,epsilon,True)
    
    contours = [cv.convexHull(c) for c in contours]
    contours = [cnt for cnt in contours if len(cnt) == 4]


    vis = original.copy()
    cv.drawContours(vis, contours, -1, (0,255,0), 2)
    cv.imshow('test', vis)
    cv.waitKey(0)


    # use hulls to capture bad color-thresholded signs
    # use minimum enclosing box, which should work even if warped perspective
    # filter by area, keep the biggest box?
    # or filter by edges, searching for squares (4 edges)
