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
    lower_green = np.array([24,40,0])
    upper_green = np.array([95, 255,255])


    # Threshold the HSV image
    mask = cv.inRange(hsv, lower_green, upper_green)

    # Bitwise-AND mask and original image
    masked = cv.bitwise_and(original, original, mask= mask)

    # perform closing operation to try to find more closed contours
    kernel = cv.getStructuringElement(cv.MORPH_RECT,(9,9))
    masked = cv.morphologyEx(masked, cv.MORPH_CLOSE, kernel)


    edges = cv.Canny(masked, 50, 200)
    contours, _ = cv.findContours(edges, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)


    # approximate contours
    for i, cnt in enumerate(contours):
        epsilon = 0.04*cv.arcLength(cnt,True)
        contours[i] = cv.approxPolyDP(cnt,epsilon,True)
    
    hulls = [cv.convexHull(c) for c in contours]

    vis = original.copy()
    cv.drawContours(vis, contours, -1, (0,255,0), 1)
    cv.imshow('test', vis)
    cv.waitKey(0)

    vis = original.copy()
    cv.drawContours(vis, hulls, -1, (0,255,0), 1)
    cv.imshow('test', vis)
    cv.waitKey(0)


    # use hulls to capture bad color-thresholded signs
    # use minimum enclosing box, which should work even if warped perspective
    # filter by area, keep the biggest box?
    # or filter by edges, searching for squares (4 edges)
