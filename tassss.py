import cv2 as cv
import numpy as np
import imutils
from task import getContours, preProcessing

small_image = cv.imread('threat_images/BAGGAGE_20170522_113049_80428_A.jpg')
large_image = cv.imread('background/S0320365070_20180821160850_L-12_5.jpg')

def Processing(image): 
    imgGray = cv.cvtColor(image,cv.COLOR_BGR2GRAY)
    imgBlur = cv.GaussianBlur(imgGray,(5,5),1)
    # for taking backgroung we have to keep 50
    imgCanny = cv.Canny(imgBlur,50,50) 
    # for taking more clearly we use below parameters
    kernel = np.ones((5,5))
    imgDial = cv.dilate(imgCanny,kernel,iterations=2)
    imgThres = cv.erode(imgDial,kernel,iterations=1)
    return imgThres

def getcon (img) :
    
    contours = cv.findContours(img,cv.RETR_EXTERNAL,cv.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(contours)
    c = max(contours, key=cv.contourArea)
    # determine the most extreme points along the contour
    extLeft = tuple(c[c[:, :, 0].argmin()][0])
    extRight = tuple(c[c[:, :, 0].argmax()][0])
    extTop = tuple(c[c[:, :, 1].argmin()][0])
    extBot = tuple(c[c[:, :, 1].argmax()][0]) 
    #cv.drawContours(img, [c], -1, (0, 255, 255), 2)
    crop_img = small_image[extTop[1]:extBot[1],extLeft[0]:extRight[0]]
    return crop_img

def getconb (img) :
    contours = cv.findContours(img,cv.RETR_EXTERNAL,cv.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(contours)
    c = max(contours, key=cv.contourArea)
    # determine the most extreme points along the contour
    extLeft = tuple(c[c[:, :, 0].argmin()][0])
    extRight = tuple(c[c[:, :, 0].argmax()][0])
    extTop = tuple(c[c[:, :, 1].argmin()][0])
    extBot = tuple(c[c[:, :, 1].argmax()][0]) 
    #cv.drawContours(img, [c], -1, (0, 255, 255), 2)
    crop_img = large_image[extTop[1]:extBot[1],extLeft[0]:extRight[0]]
    return crop_img    

def rotate_image(image, angle):
  rotated = imutils.rotate_bound(image, angle)
  l_image = np.zeros(rotated.shape[:3],dtype= np.uint8)
  l_image.fill(255)
  rotated[np.where((rotated==[0,0,0]).all(axis=2))] = [255,255,255]
  small_img_gray = cv.cvtColor(rotated, cv.COLOR_RGB2GRAY)
  
  ret, mask = cv.threshold(small_img_gray, 180, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)
  kernel = cv.getStructuringElement(cv.MORPH_RECT, (2,4))
  mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, kernel)
  #cv.imshow('rt',mask)
  mask_inv = cv.bitwise_not(mask)
  fg = cv.bitwise_and(rotated,rotated,mask = mask_inv)
  bg = cv.bitwise_or(l_image,l_image,mask=mask)
  final = cv.add(bg,fg)
  #cv.imshow('g',final)

  return rotated  

def roi (simage,limage):
    r = 100.0 / simage.shape[1]
    dim = (100, int(simage.shape[0] * r))
    #   perform the actual resizing of the image and show it
    resized = cv.resize(simage, dim, interpolation = cv.INTER_AREA)
    bimage = Processing(limage)
    conimage  = getconb(bimage)
    global x_offset
    global y_offset
    x_offset = int(conimage.shape[1]/2)
    y_offset = int(conimage.shape[0]/2)
    x_end = x_offset + resized.shape[1]
    y_end = y_offset + resized.shape[0]
    

    roi = limage[y_offset:y_end,x_offset:x_end]
    return roi , resized

def mask (simage , limage):
    small_img_gray = cv.cvtColor(simage, cv.COLOR_RGB2GRAY)
    
    #cv.imshow('g',small_img_gray)
    ret, mask = cv.threshold(small_img_gray, 200, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (2,4))
    mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, kernel)

    #cv.imshow('ma',mask)
    bg = cv.bitwise_or(limage,limage,mask=mask)
    cv.imshow('bg',bg)
    mask_inv = cv.bitwise_not(mask)  # change
    #cv.imshow('mi',mask_inv)
    fg = cv.bitwise_or(simage,simage, mask=mask_inv)
    #
    # cv.imshow('fg',fg)
    #final_roi = cv.add(bg,fg)
    final_roi = cv.addWeighted(bg,0.8,fg,0.2,0)

    return final_roi




imgThres = Processing(small_image)
crop_img = getcon(imgThres)
#cv.imshow('crop',crop_img)

rotate_image = rotate_image(crop_img,45)  
roi , resized = roi(rotate_image,large_image)
small_img = mask(resized,roi)
large_image[y_offset : y_offset + small_img.shape[0], x_offset : x_offset + small_img.shape[1]]= small_img
cv.imshow('large',large_image)


cv.waitKey(0)

