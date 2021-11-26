import cv2 as cv
import numpy as np

big_image = cv.imread('background/BAGGAGE_20180811_175323_83216_B_1.jpg')
small_image = cv.imread('threat_images/BAGGAGE_20170522_113049_80428_A.jpg')
r = 100.0 / small_image.shape[1]
dim = (100, int(small_image.shape[0] * r))
# perform the actual resizing of the image and show it
resized = cv.resize(small_image, dim, interpolation = cv.INTER_AREA)
croping = resized[80:150,2:98]
#cv.imshow('crop',croping)
#res = cv.resize(small_image,(221,100))
#cv.imshow('res',res)
#cv.imshow('Big',big_image)
#cv.imshow('small',small_image)
#cv.imshow('resized',resized)
#print(big_image.shape)
#print(resized.shape)
#print(small_image.shape)
#print(croping.shape)

x_offset = 80
y_offset = 160
rows,columns,chanels = croping.shape
roi = big_image[y_offset:230, x_offset:176]
cv.imshow('roi',roi)
#x_end = x_offset + resized.shape[1]
#y_end = y_offset + resized.shape[0]
x_end = x_offset + croping.shape[1]
y_end = y_offset + croping.shape[0]
#big_image[y_offset:y_end,x_offset:x_end] = resized
big_image[y_offset:y_end,x_offset:x_end] = croping
#cv.imshow('merge',big_image)

# mask
small_img_gray = cv.cvtColor(croping, cv.COLOR_RGB2GRAY)
cv.imshow('sm',small_img_gray)
ret, mask = cv.threshold(small_img_gray, 120, 255, cv.THRESH_BINARY)
#print(mask)
#cv.imshow('ma',mask)
print(roi.shape)
print(small_img_gray.shape)
bg = cv.bitwise_or(roi,roi,small_img_gray)
cv.imshow('bg',bg)
mask_inv = cv.bitwise_not(small_img_gray)
#cv.imshow(mask_inv)
fg = cv.bitwise_and(croping,croping, mask=mask_inv)
#cv.imshow(fg)
#final_roi = cv.add(bg,fg)
#cv.imshow(final_roi)
#small_img = final_roi
#big_image[y_offset : y_offset + small_img.shape[0], x_offset : x_offset + small_img.shape[1]]= small_img
#cv.imshow('big_image',big_image)
cv.waitKey(0)