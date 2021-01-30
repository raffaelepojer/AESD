import cv2 as cv
import numpy as np
import os
import matplotlib.pyplot as plt

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

def detectSign(target):
    # list containing the tuple (template, NAME)
    template = []
    # 0 = right
    # 1 = left
    # 2 = down
    # 3 = nothing

    # index goes from 0 to 10 (to update)
    # template.append( (cv.imread(os.path.join('dataset', 'template', 'chair-right.jpg'), 0), "CHAIR RIGHT", 0 ) )
    # template.append(  (cv.imread(os.path.join('dataset', 'template', 'chair-left.jpg'), 0), "CHARI LEFT", 1 ) )

    # template.append( (cv.imread(os.path.join('dataset', 'template', 'right-arrow-small.png'), 0), "ARROW RIGHT", 0 ) )
    # template.append( (cv.imread(os.path.join('dataset', 'template', 'left-arrow-small.png'), 0), "ARROW LEFT", 1 ) )
    template.append( (cv.imread(os.path.join('dataset', 'template', 'down-arrow-small.png'), 0), "ARROW DOWN", 2 ) )

    template.append( (cv.imread(os.path.join('dataset', 'template', 'right2.jpg'), 0), "RIGHT SIGN", 0 ) )
    template.append( (cv.imread(os.path.join('dataset', 'template', 'left2.jpg'), 0), "LEFT SIGN", 1 ) )

    template.append( (cv.imread(os.path.join('dataset', 'template', 'right-disable.jpg'), 0), "RIGHT DISABLE SIGN", 0 ) )
    template.append( (cv.imread(os.path.join('dataset', 'template', 'left-disable.jpg'), 0), "LEFT DISABLE SIGN", 1 ) )
    
    # template.append( (cv.imread(os.path.join('dataset', 'template', 'right-door-template-small.jpg'), 0), "DOOR RIGHT", 0 ) )
    # template.append( (cv.imread(os.path.join('dataset', 'template', 'left-door-template-small.jpg'), 0), "DOOR LEFT", 1 ) )

    # template.append( (cv.imread(os.path.join('dataset', 'template', 'manrun-right.jpg'), 0), "MAN RIGHT", 0 ) )
    # template.append( (cv.imread(os.path.join('dataset', 'template', 'manrun-left.jpg'), 0), "MAN LEFT", 1 ) )

    template.append( (cv.imread(os.path.join('dataset', 'template', 'right-stairs.jpg'), 0), "RIGHT STAIRS", 0 ) )
    template.append( (cv.imread(os.path.join('dataset', 'template', 'left-stairs.jpg'), 0), "LEFT STAIRS", 1 ) )

    template.append( (cv.imread(os.path.join('dataset', 'template', 'right-emergency.jpg'), 0), "RIGHT EMERGENCY", 0 ) )
    template.append( (cv.imread(os.path.join('dataset', 'template', 'left-emergency.jpg'), 0), "LEFT EMERGENCY", 1 ) )

    template.append( (cv.imread(os.path.join('dataset', 'template', 'door-down.png'), 0), "DOWN DOOR", 2) )

    template.append( (cv.imread(os.path.join('dataset', 'template', 'calm.png'), 0), "CALM", 3 ) )

    # to display the match and print the number
    DEBUG = False
    # hyperparameter to set, if there are less than this number of points the image is not detected 
    MIN_MATCH_COUNT = 12
    img2 = target
    # img2 = cv.cvtColor(img2, cv.COLOR_BGR2GRAY)
    index = 0
    # initialize detect list (nella maniera più smarza che esista, la mia conoscenza di python si riassume in queste 3 righe)
    detected = []
    for i in template:
        detected.append(0)

    for temp in template:
        # Initiate SIFT detector
        sift = cv.SIFT_create()

        # image8bit = cv.cvtColor(temp[0], cv.COLOR_BGR2GRAY)
        # find the keypoints and descriptors with SIFT
        kp1, des1 = sift.detectAndCompute(temp[0],None)
        kp2, des2 = sift.detectAndCompute(img2,None)

        # terminate if it is not possible to compute a descriptor for the image
        if des2 is None:
            print('No descriptor found for the sign')
            return detected


        FLANN_INDEX_KDTREE = 0
        index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
        search_params = dict(checks = 50)

        flann = cv.FlannBasedMatcher(index_params, search_params)

        matches = flann.knnMatch(des1,des2,k=2)

        # store all the good matches as per Lowe's ratio test.
        good = []
        for m,n in matches:
            if m.distance < 0.6*n.distance:
                good.append(m)

        if len(good)>MIN_MATCH_COUNT:
            src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
            dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)

            M, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC,5.0)

            # check if a homography can be computed
            if M is not None:
                matchesMask = mask.ravel().tolist()

                h,w = temp[0].shape[:2]
                pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)

                dst = cv.perspectiveTransform(pts,M)

                if DEBUG:
                    img2 = cv.polylines(img2,[np.int32(dst)],True,255,3, cv.LINE_AA)
                    print("Found %d points: %s" % (len(good),temp[1]))
            else:
                print('Cannot compute homography')

            # store the numbers of point detected, so if the find two opposite sign we can keep the higher
            # (# of points, LABEL, direction)
            detected[index] = (len(good), temp[1], temp[2])
        else:
            if DEBUG:
                print ("Not enough matches are found - %d/%d: NO %s" % (len(good),MIN_MATCH_COUNT,temp[1]))

            matchesMask = None

        index += 1

        if DEBUG:
            draw_params = dict(matchColor = (0,255,0), # draw matches in green color
                        singlePointColor = None,
                        matchesMask = matchesMask, # draw only inliers
                        flags = 2)
            img3 = cv.drawMatches(temp[0],kp1,img2,kp2,good,None,**draw_params)
            cv.imshow('gray', img3)
            # plt.show()
            cv.waitKey(0)

    # return a list containing the detected object found, 0 not found
    # each sign corresponds to a particular index in the list
    # see the template list to see which index each sign corresponds to
    return detected