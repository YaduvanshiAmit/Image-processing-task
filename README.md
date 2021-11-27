## Project Title --  Image processing using opencv python

## Installation and Usage 

  1. If you have previous/other manually installed (= not installed via pip) version of OpenCV installed (e.g. cv2 module in the root of Python's site-packages), remove it before installation to avoid conflicts.

  2. Make sure that your pip version is up-to-date (19.3 is the minimum supported version): pip install --upgrade pip. Check version with pip -V. For example Linux distributions ship usually with very old pip versions which cause a lot of unexpected problems especially with the manylinux format.

  3. Import the package:

     import cv2
     import numpy
     import imutils (!pip install imutils)

## Dataset Information

The dataset caontains a two sets of images - background and threat objects. Background images are the
background x-ray images of baggage that gets generated after passing through a X-ray machine at
airport. Threat images are the x-ray images of threats that are prohibited at airport while travelling.

In this image processing you have to combine this two image so that in future this image are use for making a good model.

## Coding Information

As above mention in this mainly three liabraries are used :-
1) cv2 - OpenCV is the huge open-source library for the computer vision, machine learning, and image processing and now it plays a major role in real-time operation which is very important in today's systems. By using it, one can process images and videos to identify objects, faces, or even handwriting of a human.

2) Numpy - NumPy is a Python library used for working with arrays. It also has functions for working in domain of linear algebra, fourier transform, and matrices. It is an open source project and you can use it freely.

3) imutils - This package includes a series of OpenCV + convenience functions that perform basics tasks such as translation, rotation, resizing, and skeletonization.

## Code Detail :-
Code in Submission.py

In line 5 and 6 we simple reading the file from the folder.
From line 8 to 17 we Created a simple function name Processing for simple image processing like convert the RGB to gray, gray to blur , blur to canny.
Why we use canny here - canny helps us to detect or sharp the edge of the image , here i also use dilate and erode
because they help in removing Noise or find a shape and size or structure of an object very well.

From line 19 to 31 we Created a function to crop the threat image , here first we use contours to bcz contours help joining all the continuous point along the boundary, which having same color or intensity. This very help tools for shape and object detection. By using this we detect the shape and using cv.contourArea find max area .
Then from max area use of tuple we find a extreme right point , extreme left , top and bottom point of structure . We use this point to crop the image . Like You see in code .

From line 33 to 44 we created a same function like above but for Background image here all process is same . WE use this function when we Select out Region of Interest .
WE discuss about this later.

From 46 to 52 we created a function for rotation , here this function take two parameters one is crop image which we have from last function and angle.In this function we use a imutils.rotate_bound function this function is very helpful in roating the image . Main Advantage is that in this our image is cut by due to rotation.
After that i use simple numpy to convert black pixel which come because of roation to convert them into white pixel.

From 54 to 68 we create a function for choose a Region of interest from background. Here firstly we do resize the image of our threat , we simply use resizing method. 

Why We Resize ? 

We have to select region of interest from the background image , How we know how much area we have to select , Our
 croping threat image size is big or approx to the background image , but we have to keep the threat image inside is so it compulasary to resize them , resizing also helps us in choosing the size of our ROI. 

 SO how we choose ROI for different image ?

 Thats why at above we created a function for contour for a background image. This Contour help us to know the Boundary of Bag , bcz we know we have to keep our threat inside the bag not outside ,So we have to know the boundary . What we did here we define a Global Parameter x_offest and y_offset, what value this x_offset and y_offset keep , they keep a ROI one point information , we get this point from countors boundary and we divide it by 3 so it always away from boundary . 

 Yes i know it create a problem when we trained a model using this because our model get habitat of one place , this is another case , there many more ways also we did that , but today we keep it simple.

 And what is x_end and y_end from where we get ?
 
 x_end and y_end is another set of point of ROI we get this from x_offset + shape of our resized image .

Using this all four point we get our ROI

From 70 to 81 , Heart of the code many this are there to explain this few lines of code 
Like before here also we created a function 

1) Firstly we convert the our resized image into gray by using cv2.cvtColor()
2) Then we use a cv2.threshold() , what this Thresholding technique did , in this we provide a threshold value , then each pixel value is compared with the threshold value. If the pixel value is smaller than the threshold , it is set to 0 otherwise it set to a maximum value . We use to darken the black pixel mainly . In our coding i use morphology here to get more clear outline.

After that we perform bitwise_or operation with our ROI to get our background image , till now our thread is not in actual color means it mainly a black .

So how we change color of that , so first we have to vuse cv2.bitwise_not() on masking from there we get mask_inv.

After that we use cv2.bitwise_and() with mask_inv and resize image from  this we get foreground image .

Now we get add our background image and foreground image we use here add.Weighted for blending the image by provind Alpha and Beta value.

In above all bitwise operation use Binnary operation of OR, AND,XOR ,NOT.

FRom line 86 to 90 , we only calling the function .
At line 91 we add that all to our Background image.

This all about code.

In this Image Processing a lots of things also can we done to generate a good image for our model .

For today we kept this to here only.

