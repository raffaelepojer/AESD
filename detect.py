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
    r = None

    for scale in np.linspace(0.2, 1.0, 20)[::-1]:
        dim = (int(img_gray.shape[1] * scale), int(img_gray.shape[0] * scale))
        resized = cv.resize(img_gray, dim, interpolation = cv.INTER_AREA)
        r = img_gray.shape[1] / float(resized.shape[1])
		# if the resized image is smaller than the template, then break
		# from the loop
        if resized.shape[0] < tH or resized.shape[1] < tW:
            break

        edged = cv.Canny(resized, 50, 200)
        resultR = cv.matchTemplate(edged, templateR, cv.TM_CCOEFF) # TM_CCOEFF is the best ?? 
        resultL = cv.matchTemplate(edged, templateL, cv.TM_CCOEFF)

        (_, maxValR, _, maxLocR) = cv.minMaxLoc(resultR)
        (_, maxValL, _, maxLocL) = cv.minMaxLoc(resultL)
		
        if foundR is None or maxValR > foundR[0]:
            foundR = (maxValR, maxLocR, r)
        if foundL is None or maxValL > foundL[0]:
            foundL = (maxValL, maxLocL, r)
        
        if maxValL >= maxValR:
            (_, maxLoc, r) = foundL
        else:
            (_, maxLoc, r) = foundR

    (startX, startY) = (int(maxLoc[0] * r), int(maxLoc[1] * r))
    (endX, endY) = (int((maxLoc[0] + tW) * r), int((maxLoc[1] + tH) * r))
    return (startX, startY, endX, endY)