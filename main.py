# -*- coding: utf-8 -*-

import cv2 as cv
import matplotlib.pyplot as plt

def imshow(bgr_img, title = None):
    # change color order, form bgr ot rgb
    # so that matplotlib displays the image correctly
    # this is required because OpenCV uses BGR by default
    # while matplotlib uses RGB
    plt.imshow(img[...,::-1])
    
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
img[:,:,0] = cv.equalizeHist(img[:,:,0])
img[:,:,1] = cv.equalizeHist(img[:,:,1])
img[:,:,2] = cv.equalizeHist(img[:,:,2])

plt.subplot(223)
imshow(img, 'Equalized image')

plt.subplot(224)
plt.title('Equalized histogram')

for i,col in enumerate(color):
    histr = cv.calcHist([img],[i],None,[256],[0,256])
    plt.plot(histr,color = col)
    plt.xlim([0,256])
    

plt.show()