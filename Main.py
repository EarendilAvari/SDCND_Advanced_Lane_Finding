#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr  7 02:01:03 2019

@author: earendilavari
"""

# %%%%%%%%%%%%%%%%%% IMPORT USED PACKAGES %%%%%%%%%%%%%%%%%%
import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

exec(open('Camera.py').read())
exec(open('BinaryImg.py').read())

# %%%%%%%%%%%%%%%%%% CAMERA CALIBRATION %%%%%%%%%%%%%%%%%%%%

# Read in and make a list of calibration images
# imagesPath is a list of the names of all calibration images
imagesPath = glob.glob('camera_cal/calibration*.jpg')
# cam is an object of type Camera
cam = Camera()
# The camera matrix and the coeficients are calculated 
matrix, dist = cam.Calibrate(imagesPath, 9, 6)
distortedImgTest = mpimg.imread('camera_cal/calibration1.jpg')
undistortedImgTest = cam.UndistortImage(distortedImgTest, matrix, dist)

# Plot the distorted and undistorted test image
figure1, fig1_axes = plt.subplots(1,2, figsize=(10,5))
figure1.tight_layout()
figure1.suptitle('Calibration image used. Before and after calibration')
fig1_axes[0].imshow(distortedImgTest)
fig1_axes[0].set_title('Distorted Image', fontsize = 10)
fig1_axes[1].imshow(undistortedImgTest)
fig1_axes[1].set_title('Undistorted Image', fontsize = 10)
plt.subplots_adjust(top = 1.1, bottom = 0)
figure1.savefig('ImgsReport/01_CalibImgAfterUndistort')

undistortedImgsSavePath = 'camera_cal/Undistorted/calibration%i.jpg'
cam.UndistortAndSaveImageList(imagesPath, matrix, dist, undistortedImgsSavePath)

### By plotting both images (undistorted and distorted) it can be seen how the undistorting processing
### worked on the camera. In the case of this camera, the original images are not that distorted, only in the 
### edges a convex distortion can be seen, which is corrected on the undistorted image.

# %%%%%%%%%%%%%%%%%% UNDISTORT A TEST IMAGE %%%%%%%%%%%%%%%%%%

# Now that the camera matrix and the calibration coeficients are calculated, they can be used to undistort one
# of the test images

# Let's try it with "straight_lines1.jpg'. This image will be used further for the development of the pipeline

# Reads the test image
imgStraightLines1 = mpimg.imread('test_images/straight_lines1.jpg')
imgStraightLines1Undist = cam.UndistortImage(imgStraightLines1, matrix, dist)

# Plot both images
figure2, fig2_axes = plt.subplots(1,2, figsize=(10,5))
figure2.tight_layout()
figure2.suptitle('Image "straight_lines1.jpg" before and after calibration')
fig2_axes[0].imshow(imgStraightLines1)
fig2_axes[0].set_title('Distorted Image', fontsize = 10)
fig2_axes[1].imshow(imgStraightLines1Undist)
fig2_axes[1].set_title('Undistorted Image', fontsize = 10)
plt.subplots_adjust(top = 1.1, bottom = 0)
figure2.savefig('ImgsReport/02_straight_lines1_beforeAndAfterCalib')

### By plotting this test image it can be seen that a little bit of the original image is missing on the 
### undistorted image. Also some objects are bigger, but the difference is actually minimal.

#%% Let's try it with 'test1.jpg' now.

# Reads the test image
imgTest1 = mpimg.imread('test_images/test1.jpg')
imgTest1Undist = cam.UndistortImage(imgTest1, matrix, dist)
figure3, fig3_axes = plt.subplots(1,2, figsize=(10,5))
figure3.tight_layout()
figure3.suptitle('Image "test1.jpg" before and after calibration')
fig3_axes[0].imshow(imgTest1)
fig3_axes[0].set_title('Distorted Image', fontsize = 10)
fig3_axes[1].imshow(imgTest1Undist)
fig3_axes[1].set_title('Undistorted Image', fontsize = 10)
plt.subplots_adjust(top = 1.1, bottom = 0)
figure3.savefig('ImgsReport/03_test1_beforeAndAfterCalib')

### On this image it can be seen again how in the undistorted image, the contents on the borders of the 
### images is missing and some objects are bigger.

# %%%%%%%%%%%%%%%%%% BINARY IMAGE FOR LINE DETECTION %%%%%%%%%%%%%%%%%%%%

'''
To create binary images, the class BinaryImg was created with the function GradientCalc. 
This function can create binary images in all the possible cases seen in the lessons. 

It can convert the image into grayscale, HSL or HSV. From it, it can calculate the gradient in the X direction, in the Y direction,
its magnitude or its direction. If the image is converted to HSL or HSV, one of the channels need to be selected.

By using this function, it can be easily compared between the different methods in order to get a binary image.

'''

# Lets create an object of type BinaryImg
binImg = BinaryImg()

# Now lets try the function for directionX, directionY and magnitude using a grayscale image. That will be done
# using the image 'straight_lines1.png' after being undistorted

strLines1BinGrayDirX = binImg.GradientCalc(imgStraightLines1Undist, imgType = 'grayscale', calcType = 'dirX', thresh = (40, 120))
strLines1BinGrayDirY = binImg.GradientCalc(imgStraightLines1Undist, imgType = 'grayscale', calcType = 'dirY', thresh = (40, 120))
strLines1BinGrayMag = binImg.GradientCalc(imgStraightLines1Undist, imgType = 'grayscale', calcType = 'magnitude', thresh = (40, 120))

figure4, fig4_axes = plt.subplots(2,2, figsize = (10, 8))
figure4.tight_layout()
figure4.suptitle('Binary images created with grayscale image and gradient. Threshold = (40, 120)')
fig4_axes[0, 0].imshow(imgStraightLines1Undist)
fig4_axes[0, 0].set_title('Original Image')
fig4_axes[0, 1].imshow(strLines1BinGrayDirX, cmap = 'gray')
fig4_axes[0, 1].set_title('Binary Grayscale with gradient in X direction')
fig4_axes[1, 0].imshow(strLines1BinGrayDirY, cmap = 'gray')
fig4_axes[1, 0].set_title('Binary Grayscale with gradient in Y direction')
fig4_axes[1, 1].imshow(strLines1BinGrayMag, cmap = 'gray')
fig4_axes[1, 1].set_title('Binary Grayscale with magnitude of gradient')
figure4.savefig('ImgsReport/04_BinImgsGrayGradient')


'''
By showing all three images can be seen that the lines are better shown on the binary image created by taking the gradient in the X
direction. The difference is not that big though. Also, if the threshold is increased, the rest of the scenery disapears before the
lines, because the lines have a more strong gradient.
'''

# %% Let's create now a new binary image calculating the direction of the gradient 

strLines1BinGrayDir = binImg.GradientCalc(imgStraightLines1Undist, imgType = 'grayscale', calcType = 'direction', thresh = (0.8, 1.2))

figure5, fig5_axes = plt.subplots(1, 2, figsize = (10, 5))
figure5.tight_layout()
figure5.suptitle('Binary images created with grayscale image and gradient direction. \nThreshold = (0.7 rad, 1.3 rad)')
fig5_axes[0].imshow(imgStraightLines1Undist)
fig5_axes[0].set_title('Original Image')
fig5_axes[1].imshow(strLines1BinGrayDir, cmap = 'gray')
fig5_axes[1].set_title('Binary Grayscale with direction of gradient')
figure5.savefig('ImgsReport/05_BinImgsGrayGradientDir')

'''
The binary image created by calculating the direction of the gradient shows a lot of noise, but with very good defined lines. For now 
we will discard this result, but it may be usefull in the future.
'''

# %% Now let's see what we get by using HSL color space

strLines1BinHSLchHDirX = binImg.GradientCalc(imgStraightLines1Undist, imgType = 'HSL', imgChannel = 'h', calcType = 'dirX', thresh = (40, 120))
strLines1BinHSLchHDirY = binImg.GradientCalc(imgStraightLines1Undist, imgType = 'HSL', imgChannel = 'h', calcType = 'dirY', thresh = (40, 120))
strLines1BinHSLchHMag = binImg.GradientCalc(imgStraightLines1Undist, imgType = 'HSL', imgChannel = 'h', calcType = 'magnitude', thresh = (40, 120))
strLines1BinHSLchSDirX = binImg.GradientCalc(imgStraightLines1Undist, imgType = 'HSL', imgChannel = 's', calcType = 'dirX', thresh = (40, 120))
strLines1BinHSLchSDirY = binImg.GradientCalc(imgStraightLines1Undist, imgType = 'HSL', imgChannel = 's', calcType = 'dirY', thresh = (40, 120))
strLines1BinHSLchSMag = binImg.GradientCalc(imgStraightLines1Undist, imgType = 'HSL', imgChannel = 's', calcType = 'magnitude', thresh = (40, 120))
strLines1BinHSLchLDirX = binImg.GradientCalc(imgStraightLines1Undist, imgType = 'HSL', imgChannel = 'l', calcType = 'dirX', thresh = (40, 120))
strLines1BinHSLchLDirY = binImg.GradientCalc(imgStraightLines1Undist, imgType = 'HSL', imgChannel = 'l', calcType = 'dirY', thresh = (40, 120))
strLines1BinHSLchLMag = binImg.GradientCalc(imgStraightLines1Undist, imgType = 'HSL', imgChannel = 'l', calcType = 'magnitude', thresh = (40, 120))

figure6, fig6_axes = plt.subplots(3,3, figsize = (11.5, 11))
figure6.tight_layout()
figure6.suptitle('Binary images created with HSL color space and gradient. Threshold = (40, 120)')
fig6_axes[0,0].imshow(strLines1BinHSLchHDirX, cmap = 'gray')
fig6_axes[0,0].set_title('Channel H, Direction X', fontsize = 10)
fig6_axes[0,1].imshow(strLines1BinHSLchHDirY, cmap = 'gray')
fig6_axes[0,1].set_title('Channel H, Direction Y', fontsize = 10)
fig6_axes[0,2].imshow(strLines1BinHSLchHMag, cmap = 'gray')
fig6_axes[0,2].set_title('Channel H, Magnitude', fontsize = 10)
fig6_axes[1,0].imshow(strLines1BinHSLchSDirX, cmap = 'gray')
fig6_axes[1,0].set_title('Channel S, Direction X', fontsize = 10)
fig6_axes[1,1].imshow(strLines1BinHSLchSDirY, cmap = 'gray')
fig6_axes[1,1].set_title('Channel S, Direction Y', fontsize = 10)
fig6_axes[1,2].imshow(strLines1BinHSLchSMag, cmap = 'gray')
fig6_axes[1,2].set_title('Channel S, Magnitude', fontsize = 10)
fig6_axes[2,0].imshow(strLines1BinHSLchLDirX, cmap = 'gray')
fig6_axes[2,0].set_title('Channel L, Direction X', fontsize = 10)
fig6_axes[2,1].imshow(strLines1BinHSLchLDirY, cmap = 'gray')
fig6_axes[2,1].set_title('Channel L, Direction Y', fontsize = 10)
fig6_axes[2,2].imshow(strLines1BinHSLchLMag, cmap = 'gray')
fig6_axes[2,2].set_title('Channel L, Magnitude', fontsize = 10)
plt.subplots_adjust(top = 0.9, bottom = 0.1)
figure6.savefig('ImgsReport/06_BinImgsHSLGradient')

'''
By showing all the images received can be seen that the channel H is not usefull to draw the lines into the binary images, In it a lof of scenery is detected
but the lines not. The channel S does a very good job by detecting the lines, but they are missing sometimes. There is not a big difference between taking 
the gradient in direction x, y or the magnitude.
On the other hand, the channel L does even a better job than the channel S by detecting the lines, but it also detects some scenery. It must be remembered that
all that scenery will not be considered when the line pixels will be detected and calculated, so it could be safely said that the L channel is here the best 
option to detect the lines. 
'''

# %% Only to try, lets see what we get with the same settings but another theshold range

strLines1BinHSLchHDirX = binImg.GradientCalc(imgStraightLines1Undist, imgType = 'HSL', imgChannel = 'h', calcType = 'dirX', thresh = (30, 150))
strLines1BinHSLchHDirY = binImg.GradientCalc(imgStraightLines1Undist, imgType = 'HSL', imgChannel = 'h', calcType = 'dirY', thresh = (30, 150))
strLines1BinHSLchHMag = binImg.GradientCalc(imgStraightLines1Undist, imgType = 'HSL', imgChannel = 'h', calcType = 'magnitude', thresh = (30, 150))
strLines1BinHSLchSDirX = binImg.GradientCalc(imgStraightLines1Undist, imgType = 'HSL', imgChannel = 's', calcType = 'dirX', thresh = (30, 150))
strLines1BinHSLchSDirY = binImg.GradientCalc(imgStraightLines1Undist, imgType = 'HSL', imgChannel = 's', calcType = 'dirY', thresh = (30, 150))
strLines1BinHSLchSMag = binImg.GradientCalc(imgStraightLines1Undist, imgType = 'HSL', imgChannel = 's', calcType = 'magnitude', thresh = (30, 150))
strLines1BinHSLchLDirX = binImg.GradientCalc(imgStraightLines1Undist, imgType = 'HSL', imgChannel = 'l', calcType = 'dirX', thresh = (30, 150))
strLines1BinHSLchLDirY = binImg.GradientCalc(imgStraightLines1Undist, imgType = 'HSL', imgChannel = 'l', calcType = 'dirY', thresh = (30, 150))
strLines1BinHSLchLMag = binImg.GradientCalc(imgStraightLines1Undist, imgType = 'HSL', imgChannel = 'l', calcType = 'magnitude', thresh = (30, 150))

figure7, fig7_axes = plt.subplots(3,3, figsize = (11.5, 11))
figure7.tight_layout()
figure7.suptitle('Binary images created with HSL color space and gradient. Threshold = (30, 150)')
fig7_axes[0,0].imshow(strLines1BinHSLchHDirX, cmap = 'gray')
fig7_axes[0,0].set_title('Channel H, Direction X', fontsize = 10)
fig7_axes[0,1].imshow(strLines1BinHSLchHDirY, cmap = 'gray')
fig7_axes[0,1].set_title('Channel H, Direction Y', fontsize = 10)
fig7_axes[0,2].imshow(strLines1BinHSLchHMag, cmap = 'gray')
fig7_axes[0,2].set_title('Channel H, Magnitude', fontsize = 10)
fig7_axes[1,0].imshow(strLines1BinHSLchSDirX, cmap = 'gray')
fig7_axes[1,0].set_title('Channel S, Direction X', fontsize = 10)
fig7_axes[1,1].imshow(strLines1BinHSLchSDirY, cmap = 'gray')
fig7_axes[1,1].set_title('Channel S, Direction Y', fontsize = 10)
fig7_axes[1,2].imshow(strLines1BinHSLchSMag, cmap = 'gray')
fig7_axes[1,2].set_title('Channel S, Magnitude', fontsize = 10)
fig7_axes[2,0].imshow(strLines1BinHSLchLDirX, cmap = 'gray')
fig7_axes[2,0].set_title('Channel L, Direction X', fontsize = 10)
fig7_axes[2,1].imshow(strLines1BinHSLchLDirY, cmap = 'gray')
fig7_axes[2,1].set_title('Channel L, Direction Y', fontsize = 10)
fig7_axes[2,2].imshow(strLines1BinHSLchLMag, cmap = 'gray')
fig7_axes[2,2].set_title('Channel L, Magnitude', fontsize = 10)
plt.subplots_adjust(top = 0.9, bottom = 0.1)
figure7.savefig('ImgsReport/07_BinImgsHSLGradientOtherThreshold')

'''
By increasing the threshold range, it gets more evident that the Channel L, with the gradient in the X direction is the best approach to detect the lines. 
They are drawn very clear and with not that many other irrelevant elements. The binary images created with the channel S are still missing some parts of the 
line.

This can be explained by thinking how the colorspace HSL works. The H component (Hue) corresponds to the value of the base color 
(red, green, blue, cyan, magenta and yellow), the S component (Saturation) corresponds to the strength of the shown color, and the L (Lightness) corresponds to
how close the color is to white. Since the lane lines are normally white or light yellow, they are more easy to detect using the L component because they are 
very close to white. But what if the line is darker yellow? In that case the channel L can fail to detect the lane lines, but in the channel S, they will still 
be strong enough.
'''






