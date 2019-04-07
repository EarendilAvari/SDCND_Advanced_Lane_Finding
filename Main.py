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
figure1, (ax1, ax2) = plt.subplots(1,2, figsize=(24,9))
figure1.tight_layout()
ax1.imshow(distortedImgTest)
ax1.set_title('Distorted Image', fontsize = 40)
ax2.imshow(undistortedImgTest)
ax2.set_title('Undistorted Image', fontsize = 40)
plt.subplots_adjust(left = 0.03, right = 1, top = 0.9, bottom = 0.)

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
figure2, (ax3, ax4) = plt.subplots(1,2, figsize=(24,9))
figure2.tight_layout()
ax3.imshow(imgStraightLines1)
ax3.set_title('Distorted Image', fontsize = 40)
ax4.imshow(imgStraightLines1Undist)
ax4.set_title('Undistorted Image', fontsize = 40)
plt.subplots_adjust(left = 0.03, right = 1, top = 0.9, bottom = 0.)

### By plotting this test image it can be seen that a little bit of the original image is missing on the 
### undistorted image. Also some objects are bigger, but the difference is actually minimal.

#%% Let's try it with 'test1.jpg' now.

# Reads the test image
imgTest1 = mpimg.imread('test_images/test1.jpg')
imgTest1Undist = cam.UndistortImage(imgTest1, matrix, dist)
figure3, (ax5, ax6) = plt.subplots(1,2, figsize=(24,9))
figure3.tight_layout()
ax5.imshow(imgTest1)
ax5.set_title('Distorted Image', fontsize = 40)
ax6.imshow(imgTest1Undist)
ax6.set_title('Undistorted Image', fontsize = 40)
plt.subplots_adjust(left = 0.03, right = 1, top = 0.9, bottom = 0.)

### On this image it can be seen again how in the undistorted image, the contents on the borders of the 
### images is missing and some objects are bigger.



