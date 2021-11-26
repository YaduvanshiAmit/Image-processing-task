import cv2 as cv
import numpy as np
import imutils
from task import getContours

small_image = cv.imread('threat_images/BAGGAGE_20170522_113049_80428_A.jpg')

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

def rotate_image(image, angle):
  rotated = imutils.rotate_bound(image, angle)
  return rotated   

imgThres = Processing(small_image)
crop_img = getcon(imgThres)
rotate_image = rotate_image(crop_img,45)  
cv.imshow('rota',rotate_image)
cv.waitKey(0)