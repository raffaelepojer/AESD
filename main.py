# -*- coding: utf-8 -*-

import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import os

def imshow(bgr_img, title = None):
    # change color order, form bgr ot rgb
    # so that matplotlib displays the image correctly
    # this is required because OpenCV uses BGR by default
    # while matplotlib uses RGB
    plt.imshow(bgr_img[...,::-1])
    
    if title is not None:
        plt.title(title)
    
    plt.axis('off')


def histPlot(bgr_img, title = None):
    # compute and plot histogram

    color = ('b','g','r')

    for i,col in enumerate(color):
        histr = cv.calcHist([bgr_img],[i],None,[256],[0,256])
        plt.plot(histr,color = col)
        plt.xlim([0,256])
    
    if title is not None:
        plt.title(title)


img = cv.imread(os.path.join('dataset', 'image17.jpg'))


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


# plot_dim = 210
# plt.subplot(plot_dim+1)
# imshow(img, 'Original image')
# plt.subplot(plot_dim+2)
# imshow(equ2, 'Adaptive histogram equalization')
# plt.show()


# conversion to HSV
hsv = cv.cvtColor(equ2, cv.COLOR_BGR2HSV)

# color thresholding

# range of green in the dataset
lower_green = np.array([34,23,0])
upper_green = np.array([98, 255,160])

# Possible alternative mask, more strict
# we lose details but we have a bit less noise
# lower_green2 = np.array([48,23,0])
# upper_green2= np.array([89, 255,160])

# Threshold the HSV image
mask = cv.inRange(hsv, lower_green, upper_green)

# Bitwise-AND mask and original image
masked = cv.bitwise_and(img, img, mask= mask)


edges = cv.Canny(masked, 100, 200)


contours, hierarchy = cv.findContours(edges, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

vis = img.copy()
cv.drawContours(vis, contours, -1, (0,255,0), 1)
imshow(vis)

plt.show()

