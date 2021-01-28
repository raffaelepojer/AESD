import cv2 as cv
import numpy as np
import os
from matplotlib import pyplot as plt

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

    for scale in np.linspace(0.01, 1.0, 30)[::-1]:
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
        print("Max value arrow: ", np.amax(res))
        (startX, startY) = (int(maxLoc[0] * r1), int(maxLoc[1] * r0))
        (endX, endY) = (int((maxLoc[0] + tW) * r1), int((maxLoc[1] + tH) * r0))
        return (startX, startY, endX, endY, det)
    else:
        print("No arrow found")
        return (0, 0, 0, 0, "NO_ARROW")

def findDoor(target):
    img_gray = cv.cvtColor(target, cv.COLOR_BGR2GRAY)

    templateR = cv.imread(os.path.join('dataset', 'template', 'right-door-template-small.jpg'), cv.IMREAD_COLOR)
    templateL = cv.imread(os.path.join('dataset', 'template', 'left-door-template-small.jpg'), cv.IMREAD_COLOR)
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

    for scale in np.linspace(0.01, 1.0, 30)[::-1]:
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
        det = "LEFT_DOOR"
        res = resultL
    else:
        (_, maxLoc, r0, r1) = foundR
        res = resultR
        det = "RIGHT_DOOR"

    threshold = 0
    if np.amax(res) > threshold:
        print("Max value door: ", np.amax(res))
        (startX, startY) = (int(maxLoc[0] * r1), int(maxLoc[1] * r0))
        (endX, endY) = (int((maxLoc[0] + tW) * r1), int((maxLoc[1] + tH) * r0))
        return (startX, startY, endX, endY, det)
    else:
        print("No door found")
        return (0, 0, 0, 0, "NO_DOOR")

def findChair(target):
    img_gray = cv.cvtColor(target, cv.COLOR_BGR2GRAY)

    templateR = cv.imread(os.path.join('dataset', 'template', 'chair-right.jpg'), cv.IMREAD_COLOR)
    templateL = cv.imread(os.path.join('dataset', 'template', 'chair-left.jpg'), cv.IMREAD_COLOR)
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

    for scale in np.linspace(0.01, 1.0, 30)[::-1]:
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
        det = "LEFT_CHAIR"
        res = resultL
    else:
        (_, maxLoc, r0, r1) = foundR
        res = resultR
        det = "RIGHT_CHAIR"

    threshold = 0
    if np.amax(res) > threshold:
        print("Max value chair: ", np.amax(res))
        (startX, startY) = (int(maxLoc[0] * r1), int(maxLoc[1] * r0))
        (endX, endY) = (int((maxLoc[0] + tW) * r1), int((maxLoc[1] + tH) * r0))
        return (startX, startY, endX, endY, det)
    else:
        print("No chair found")
        return (0, 0, 0, 0, "NO_CHAIR")

def findMan(target):
    img_gray = cv.cvtColor(target, cv.COLOR_BGR2GRAY)

    templateR = cv.imread(os.path.join('dataset', 'template', 'manrun-right.jpg'), cv.IMREAD_COLOR)
    templateL = cv.imread(os.path.join('dataset', 'template', 'manrun-left.jpg'), cv.IMREAD_COLOR)
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

    cv.imshow("canny template", templateL)

    method = cv.TM_CCOEFF_NORMED

    for scale in np.linspace(0.01, 1.0, 30)[::-1]:
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

        # cv.imshow("canny sign", edged)
        # cv.waitKey(0)

        (minValR, maxValR, minLocR, maxLocR) = cv.minMaxLoc(resultR)
        (minValL, maxValL, minLocL, maxLocL) = cv.minMaxLoc(resultL)
	
        if foundR is None or maxValR > foundR[0]:
            foundR = (maxValR, maxLocR, r0, r1)
        if foundL is None or maxValL > foundL[0]:
            foundL = (maxValL, maxLocL, r0, r1)

    if foundL[0] >= foundR[0]:
        (_, maxLoc, r0, r1) = foundL
        det = "LEFT_MAN"
        res = resultL
    else:
        (_, maxLoc, r0, r1) = foundR
        res = resultR
        det = "RIGHT_MAN"

    threshold = 0
    if np.amax(res) > threshold:
        print("Max value man: ", np.amax(res))
        (startX, startY) = (int(maxLoc[0] * r1), int(maxLoc[1] * r0))
        (endX, endY) = (int((maxLoc[0] + tW) * r1), int((maxLoc[1] + tH) * r0))
        return (startX, startY, endX, endY, det)
    else:
        print("No chair found")
        return (0, 0, 0, 0, "NO_MAN")

def findGen(target):
    img_gray = cv.cvtColor(target, cv.COLOR_BGR2GRAY)

    templateAR = cv.imread(os.path.join('dataset', 'template', 'right-arrow-small.jpg'), cv.IMREAD_COLOR)
    templateAL = cv.imread(os.path.join('dataset', 'template', 'left-arrow-small.jpg'), cv.IMREAD_COLOR)
    templateDR = cv.imread(os.path.join('dataset', 'template', 'right-door-template-small.jpg'), cv.IMREAD_COLOR)
    templateDL = cv.imread(os.path.join('dataset', 'template', 'left-door-template-small.jpg'), cv.IMREAD_COLOR)
    templateCR = cv.imread(os.path.join('dataset', 'template', 'chair-right.jpg'), cv.IMREAD_COLOR)
    templateCL = cv.imread(os.path.join('dataset', 'template', 'chair-left.jpg'), cv.IMREAD_COLOR)
    templateMR = cv.imread(os.path.join('dataset', 'template', 'manrun-right.jpg'), cv.IMREAD_COLOR)
    templateML = cv.imread(os.path.join('dataset', 'template', 'manrun-left.jpg'), cv.IMREAD_COLOR)

    templateAR = cv.cvtColor(templateAR, cv.COLOR_BGR2GRAY)
    templateAL = cv.cvtColor(templateAL, cv.COLOR_BGR2GRAY)
    templateAR = cv.Canny(templateAR, 50, 200)
    templateAL = cv.Canny(templateAL, 50, 200)
    (tH, tW) = templateAR.shape[:2]


    foundR = None
    foundL = None
    maxLoc = None
    r0 = None
    r1 = None
    res = None
    resultL = None
    resultR = None
    det = ''

    cv.imshow("canny template", templateL)

    method = cv.TM_CCOEFF_NORMED

    for scale in np.linspace(0.01, 1.0, 30)[::-1]:
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

        # cv.imshow("canny sign", edged)
        # cv.waitKey(0)

        (minValR, maxValR, minLocR, maxLocR) = cv.minMaxLoc(resultR)
        (minValL, maxValL, minLocL, maxLocL) = cv.minMaxLoc(resultL)
	
        if foundR is None or maxValR > foundR[0]:
            foundR = (maxValR, maxLocR, r0, r1)
        if foundL is None or maxValL > foundL[0]:
            foundL = (maxValL, maxLocL, r0, r1)

    if foundL[0] >= foundR[0]:
        (_, maxLoc, r0, r1) = foundL
        det = "LEFT_MAN"
        res = resultL
    else:
        (_, maxLoc, r0, r1) = foundR
        res = resultR
        det = "RIGHT_MAN"

    threshold = 0
    if np.amax(res) > threshold:
        print("Max value man: ", np.amax(res))
        (startX, startY) = (int(maxLoc[0] * r1), int(maxLoc[1] * r0))
        (endX, endY) = (int((maxLoc[0] + tW) * r1), int((maxLoc[1] + tH) * r0))
        return (startX, startY, endX, endY, det)
    else:
        print("No chair found")
        return (0, 0, 0, 0, "NO_MAN")

def findChairHom(target):
    templateR = cv.imread(os.path.join('dataset', 'template', 'chair-right.jpg'), cv.IMREAD_COLOR)
    templateL = cv.imread(os.path.join('dataset', 'template', 'chair-left.jpg'), cv.IMREAD_COLOR)
    img_object = templateL
    img_scene = target

    MIN_MATCH_COUNT = 10
    img1 = templateL
    img2 = target

    # Initiate SIFT detector
    sift = cv.SIFT()

    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(img1,None)
    kp2, des2 = sift.detectAndCompute(img2,None)

    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks = 50)

    flann = cv.FlannBasedMatcher(index_params, search_params)

    matches = flann.knnMatch(des1,des2,k=2)

    # store all the good matches as per Lowe's ratio test.
    good = []
    for m,n in matches:
        if m.distance < 0.7*n.distance:
            good.append(m)

    if len(good)>MIN_MATCH_COUNT:
        src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
        dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)

        M, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC,5.0)
        matchesMask = mask.ravel().tolist()

        h,w = img1.shape
        pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
        dst = cv.perspectiveTransform(pts,M)

        img2 = cv.polylines(img2,[np.int32(dst)],True,255,3, cv.LINE_AA)
    else:
        print ("Not enough matches are found - %d/%d" % (len(good),MIN_MATCH_COUNT))
        matchesMask = None
    
    draw_params = dict(matchColor = (0,255,0), # draw matches in green color
                   singlePointColor = None,
                   matchesMask = matchesMask, # draw only inliers
                   flags = 2)

    img3 = cv.drawMatches(img1,kp1,img2,kp2,good,None,**draw_params)

    plt.imshow(img3, 'gray'),plt.show()