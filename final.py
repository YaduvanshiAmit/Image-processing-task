import cv2 as cv
import numpy as np

large_image = cv.imread('background/BAGGAGE_20180811_175323_83216_B_1.jpg')
smal_image = cv.imread('threat_images/BAGGAGE_20170522_113049_80428_A.jpg')
print(large_image.shape)
print(smal_image.shape)
small_image = cv.resize(smal_image,(232,70))
cv.imshow('lm',smal_image)
cv.imshow('sm',small_image)
crop = smal_image[200:332,0:232]
cv.imshow('cm',crop)
cv.waitKey(0)

x_offset = 400
y_offset = 170 

