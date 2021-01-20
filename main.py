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


img = cv.imread(os.path.join('dataset', 'image2.jpg'))


# histogram equalization to improve contrast
# equ1 = np.zeros(img.shape, dtype=np.uint8)

# equ1[:,:,0] = cv.equalizeHist(img[:,:,0])
# equ1[:,:,1] = cv.equalizeHist(img[:,:,1])
# equ1[:,:,2] = cv.equalizeHist(img[:,:,2])
    

# adaptive histogram equalization using CLAHE algorithm
equ2 = np.zeros(img.shape, dtype=np.uint8)

clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))

equ2[:,:,0] = clahe.apply(img[:,:,0])
equ2[:,:,1] = clahe.apply(img[:,:,1])
equ2[:,:,2] = clahe.apply(img[:,:,2])


plot_dim = 210
plt.subplot(plot_dim+1)
imshow(img, 'Original image')
plt.subplot(plot_dim+2)
imshow(equ2, 'Adaptive histogram equalization')
plt.show()