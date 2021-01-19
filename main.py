# -*- coding: utf-8 -*-

import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

def imshow(bgr_img, title = None):
    # change color order, form bgr ot rgb
    # so that matplotlib displays the image correctly
    # this is required because OpenCV uses BGR by default
    # while matplotlib uses RGB
    plt.imshow(bgr_img[...,::-1])
    
    if title is not None:
        plt.title(title)
    
    plt.axis('off')


img = cv.imread('dataset\\image1.jpg')


# original image and histogram
plt.subplot(221)
imshow(img, 'Original image')

plt.subplot(222)
plt.title('Original histogram')
color = ('b','g','r')
for i,col in enumerate(color):
    histr = cv.calcHist([img],[i],None,[256],[0,256])
    plt.plot(histr,color = col)
    plt.xlim([0,256])


# histogram equalization to improve constrast
equ1 = np.zeros(img.shape, dtype=np.uint8)

equ1[:,:,0] = cv.equalizeHist(img[:,:,0])
equ1[:,:,1] = cv.equalizeHist(img[:,:,1])
equ1[:,:,2] = cv.equalizeHist(img[:,:,2])

plt.subplot(223)
imshow(equ1, 'Equalized image')

plt.subplot(224)
plt.title('Equalized histogram')

for i,col in enumerate(color):
    histr = cv.calcHist([equ1],[i],None,[256],[0,256])
    plt.plot(histr,color = col)
    plt.xlim([0,256])
    

plt.show()
