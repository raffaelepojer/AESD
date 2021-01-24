# -*- coding: utf-8 -*-

import cv2 as cv
import numpy as np
import os
import roi


img = cv.imread(os.path.join('dataset', 'image5.jpg'))


# histogram equalization to improve contrast
equ1 = np.zeros(img.shape, dtype=np.uint8)

equ1[:,:,0] = cv.equalizeHist(img[:,:,0])
equ1[:,:,1] = cv.equalizeHist(img[:,:,1])
equ1[:,:,2] = cv.equalizeHist(img[:,:,2])
    

# adaptive histogram equalization using CLAHE algorithm
equ2 = np.zeros(img.shape, dtype=np.uint8)

clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))

equ2[:,:,0] = clahe.apply(img[:,:,0])
equ2[:,:,1] = clahe.apply(img[:,:,1])
equ2[:,:,2] = clahe.apply(img[:,:,2])


roi.findRoi(equ2, img)