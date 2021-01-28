import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt


def findRoi(img):
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


    # perform closing operation to try to find more closed contours
    kernel = cv.getStructuringElement(cv.MORPH_RECT,(15,15))
    mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, kernel)


    edges = cv.Canny(mask, 50, 200)
    contours, _ = cv.findContours(edges, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)


    # approximate contours
    for i, cnt in enumerate(contours):
        epsilon = 0.08*cv.arcLength(cnt,True)
        contours[i] = cv.approxPolyDP(cnt,epsilon,True)
    
    # convexity correction
    contours = [cv.convexHull(c) for c in contours]

    # filter by number of edges
    contours = [cnt for cnt in contours if len(cnt) == 4]

    # filter by area
    contours = [cnt for cnt in contours if cv.contourArea(cnt) >= 5000]

    for cnt in contours:
        print(cv.contourArea(cnt))

    return contours