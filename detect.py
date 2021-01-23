import cv2 as cv
import numpy as np
import os

'''
    Return the coordinate for the bounding box
'''
def findArrow(target):
    img_gray = cv.cvtColor(target, cv.COLOR_BGR2GRAY)

    templateR = cv.imread(os.path.join('dataset', 'template', 'right-arrow-small.jpg'), cv.IMREAD_COLOR)
    templateL = cv.imread(os.path.join('dataset', 'template', 'left-arrow-small.jpg'), cv.IMREAD_COLOR)
    templateR = cv.cvtColor(templateR, cv.COLOR_BGR2GRAY)
    templateL = cv.cvtColor(templateL, cv.COLOR_BGR2GRAY)
    templateR = cv.Canny(templateR, 50, 200)
    templateL = cv.Canny(templateL, 50, 200)
    (tH, tW) = templateR.shape[:2]
    foundR = None
    foundL = None
    maxLoc = None
    r0 = None
    r1 = None
    res = None
    resultL = None
    resultR = None
    det = ''

    method = cv.TM_CCOEFF_NORMED

    for scale in np.linspace(0.1, 1.0, 20)[::-1]:
        dim = (int(img_gray.shape[1] * scale), int(img_gray.shape[0] * scale))
        resized = cv.resize(img_gray, dim, interpolation = cv.INTER_AREA)
        r0 = img_gray.shape[0] / float(resized.shape[0])
        r1 = img_gray.shape[1] / float(resized.shape[1])
		# if the resized image is smaller than the template, then break
		# from the loop
        if resized.shape[0] < tH or resized.shape[1] < tW:
            break

        edged = cv.Canny(resized, 50, 200)
        resultR = cv.matchTemplate(edged, templateR, method) 
        resultL = cv.matchTemplate(edged, templateL, method)

        (minValR, maxValR, minLocR, maxLocR) = cv.minMaxLoc(resultR)
        (minValL, maxValL, minLocL, maxLocL) = cv.minMaxLoc(resultL)
	
        if foundR is None or maxValR > foundR[0]:
            foundR = (maxValR, maxLocR, r0, r1)
        if foundL is None or maxValL > foundL[0]:
            foundL = (maxValL, maxLocL, r0, r1)

    if foundL[0] >= foundR[0]:
        (_, maxLoc, r0, r1) = foundL
        det = "LEFT_ARROW"
        res = resultL
    else:
        (_, maxLoc, r0, r1) = foundR
        res = resultR
        det = "RIGHT_ARROW"

    threshold = 0
    if np.amax(res) > threshold:
        print("Max value: ", np.amax(res))
        (startX, startY) = (int(maxLoc[0] * r1), int(maxLoc[1] * r0))
        (endX, endY) = (int((maxLoc[0] + tW) * r1), int((maxLoc[1] + tH) * r0))
        return (startX, startY, endX, endY, det)
    else:
        print("No arrow found")
        return (0, 0, 0, 0, "NO_ARROW")