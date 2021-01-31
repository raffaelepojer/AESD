import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt


def preprocess(img):
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

    return blur


def findRoi(img):
    # conversion to HSV
    hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)

    # color thresholding

    # range of green in the dataset
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

    return contours


def order_points(pts):
    # list of coordinates orderd as:
    # top-lef, top-right, bottom-right, bottom-left
    rect = np.zeros((4, 2), dtype = "float32")

    # the top-left point will have the smallest sum
    # the bottom-right point will have the largest sum
    s = pts.sum(axis = 1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    # compute the difference between the points
    # top-right point will have the smallest difference
    # bottom-left will have the largest difference
    diff = np.diff(pts, axis = 1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    # return the ordered coordinates
    return rect


def correctPerspective(contours, img):

    signs = []

    for cnt in contours:
        pts = cnt.reshape(4, 2)
        
        # rect = tl, tr, br, bl
        rect = order_points(pts)
        (tl, tr, br, bl) = rect

        # compute destination dimensions

        # compute width of the warped image
        # maximum distance between bottom-right and bottom-left
        # x-coordiates or the top-right and top-left x-coordinates
        widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
        widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
        maxWidth = max(int(widthA), int(widthB))

        # compute height of the warped image
        # maximum distance between the top-right and bottom-right
        # y-coordinates or the top-left and bottom-left y-coordinates
        heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
        heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
        maxHeight = max(int(heightA), int(heightB))

        # compute destination box coordinates
        dst = np.array([
            [0, 0],
            [maxWidth - 1, 0],
            [maxWidth - 1, maxHeight - 1],
            [0, maxHeight - 1]], dtype = "float32")
        

        # compute the perspective transform matrix and then apply it
        M = cv.getPerspectiveTransform(rect, dst)
        warped = cv.warpPerspective(img, M, (maxWidth, maxHeight))

        signs.append(warped)

    return signs
