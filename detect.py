import cv2 as cv
import numpy as np
import os


def findArrow(target):
    # use template matching to look for arrows in the sign
    # because template matching is not scale-invariant
    # we progressively scale the image and keep the best match

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

        (_, maxValR, _, maxLocR) = cv.minMaxLoc(resultR)
        (_, maxValL, _, maxLocL) = cv.minMaxLoc(resultL)
	
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


def detectSign(target):
    # list containing the tuple (template, NAME)
    template = [
        # directions
        # 0 = right
        # 1 = left
        # 2 = down
        # 3 = nothing

        # index goes from 0 to 10 (to update)
        (cv.imread(os.path.join('dataset', 'template', 'down-arrow-small.png'), 0), "ARROW DOWN", 2 ),
        (cv.imread(os.path.join('dataset', 'template', 'right.jpg'), 0), "RIGHT SIGN", 0 ) ,
        (cv.imread(os.path.join('dataset', 'template', 'left.jpg'), 0), "LEFT SIGN", 1 ),
        (cv.imread(os.path.join('dataset', 'template', 'down.png'), 0), "DOWN SIGN", 1 ),
        (cv.imread(os.path.join('dataset', 'template', 'right-hand.jpg'), 0), "RIGHT HANDICAPPED SIGN", 0 ),
        (cv.imread(os.path.join('dataset', 'template', 'left-hand.jpg'), 0), "LEFT HANDICAPPED SIGN", 1 ),
        (cv.imread(os.path.join('dataset', 'template', 'right-stairs.jpg'), 0), "RIGHT STAIRS", 0 ),
        (cv.imread(os.path.join('dataset', 'template', 'left-stairs.jpg'), 0), "LEFT STAIRS", 1 ),
        (cv.imread(os.path.join('dataset', 'template', 'right-emergency.jpg'), 0), "RIGHT EMERGENCY", 0 ),
        (cv.imread(os.path.join('dataset', 'template', 'left-emergency.jpg'), 0), "LEFT EMERGENCY", 1 ),
        (cv.imread(os.path.join('dataset', 'template', 'door-down.png'), 0), "DOWN DOOR", 2),
        (cv.imread(os.path.join('dataset', 'template', 'calm.png'), 0), "CALM", 3 )
    ]

    # to display the match and print the number
    DEBUG = False

    # hyperparameter to set, if there are less than this number of points the image is not detected 
    MIN_MATCH_COUNT = 12

    # knn
    K = 2
    
    detected = []
    for _ in template:
        detected.append(0)

    # Initiate SIFT detector
    sift = cv.SIFT_create()
    
    # find the keypoints and descriptors with SIFT
    kp2, des2 = sift.detectAndCompute(target,None)

    for index, temp in enumerate(template):

        # find the keypoints and descriptors with SIFT
        kp1, des1 = sift.detectAndCompute(temp[0],None)

        # terminate if it is not possible to compute a descriptor for the image
        if des2 is None:
            print('No descriptor found for the sign')
            return detected

        # terminate if there are not enough keypoints for knn
        if len(kp2) < K:
            print('Not enough keypoints')
            return detected

        # flann matcher and its parameters
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
        search_params = dict(checks = 50)

        flann = cv.FlannBasedMatcher(index_params, search_params)

        # match points
        matches = flann.knnMatch(des1,des2,k=K)

        # store all the good matches as per Lowe's ratio test.
        good = []
        for m,n in matches:
            if m.distance < 0.6*n.distance:
                good.append(m)

        if len(good)>MIN_MATCH_COUNT:
            # visualize the matches
            if DEBUG:
                src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
                dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)

                M, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC,5.0)

                # check if a homography can be computed
                if M is not None:
                    matchesMask = mask.ravel().tolist()

                    h,w = temp[0].shape[:2]
                    pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)

                    dst = cv.perspectiveTransform(pts,M)

                    target = cv.polylines(target,[np.int32(dst)],True,255,3, cv.LINE_AA)
                    print("Found %d points: %s" % (len(good),temp[1]))
                else:
                    print('Cannot compute homography')

            # store the numbers of point detected, so if we find two opposite sign we can keep the higher
            # (# of points, LABEL, direction)
            detected[index] = (len(good), temp[1], temp[2])
        else:
            if DEBUG:
                print ("Not enough matches are found - %d/%d: NO %s" % (len(good),MIN_MATCH_COUNT,temp[1]))

            matchesMask = None


        if DEBUG:
            draw_params = dict(matchColor = (0,255,0), # draw matches in green color
                        singlePointColor = None,
                        matchesMask = matchesMask, # draw only inliers
                        flags = 2)
            vis = cv.drawMatches(temp[0],kp1,target,kp2,good,None,**draw_params)
            cv.imshow('Matches', vis)
            cv.waitKey(0)

    # return a list containing the detected object found, 0 not found
    # each sign corresponds to a particular index in the list
    # see the template list to see which index each sign corresponds to
    return detected