import cv2 as cv
import numpy as np

image = cv.imread('threat_images/BAGGAGE_20170522_113049_80428_A.jpg')
# we need to keep in mind aspect ratio so the image does
# not look skewed or distorted -- therefore, we calculate
# the ratio of the new image to the old image
r = 100.0 / image.shape[1]
dim = (100, int(image.shape[0] * r))
# perform the actual resizing of the image and show it
resized = cv.resize(image, dim, interpolation = cv.INTER_AREA)


def preProcessing(image): 
    imgGray = cv.cvtColor(resized,cv.COLOR_BGR2GRAY)
    imgBlur = cv.GaussianBlur(imgGray,(5,5),1)
    # for taking backgroung we have to keep 50
    imgCanny = cv.Canny(imgBlur,50,50) 
    # for taking more clearly we use below parameters
    #kernel = np.ones((5,5))
    #imgDial = cv.dilate(imgCanny,kernel,iterations=2)
    #imgThres = cv.erode(imgDial,kernel,iterations=1)
    return imgCanny

def getContours(img):
   
    contours,hierarchy = cv.findContours(img,cv.RETR_EXTERNAL,cv.CHAIN_APPROX_NONE)
    cv.drawContours(img,[contours],-1,(36, 255, 12), 2)

# Rotation
def rotate(img, angle, rotPoint=None):
    (height,width) = img.shape[:2]

    if rotPoint is None:
        rotPoint = (width//2,height//2)
    
    rotMat = cv.getRotationMatrix2D(rotPoint, angle, 1.0)
    dimensions = (width,height)

    return cv.warpAffine(img, rotMat, dimensions)

rotated = rotate(resized, 45)
#cv.imshow('Rotated', rotated)    

imgThres = preProcessing(resized)
contours,hierarchy = cv.findContours(imgThres,cv.RETR_EXTERNAL,cv.CHAIN_APPROX_NONE)
cv.drawContours(resized, contours, -1, (36, 255, 12), 2)

#imgContor = getContours(imgThres)
#cv.imshow('Cat',resized)
cv.waitKey(0)