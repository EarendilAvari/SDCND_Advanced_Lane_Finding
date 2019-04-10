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
exec(open('ProcessLines.py').read())

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

# %% Lets visualize the last idea about the HSL colorspace by ploting the same image in their HSL components

imgStraightLines1UndistHSL = cv2.cvtColor(imgStraightLines1Undist, cv2.COLOR_RGB2HLS)

figure8, fig8_axes = plt.subplots(2,2, figsize = (10, 5))
figure8.tight_layout()
figure8.suptitle('Image "straight_lines1.jpg" and its HSL components')
fig8_axes[0,0].imshow(imgStraightLines1Undist)
fig8_axes[0,0].set_title('Original image', fontsize = 10)
fig8_axes[0,1].imshow(imgStraightLines1UndistHSL[:,:,0], cmap = 'gray')
fig8_axes[0,1].set_title('Component H', fontsize = 10)
fig8_axes[1,0].imshow(imgStraightLines1UndistHSL[:,:,1], cmap = 'gray')
fig8_axes[1,0].set_title('Component L', fontsize = 10)
fig8_axes[1,1].imshow(imgStraightLines1UndistHSL[:,:,2], cmap = 'gray')
fig8_axes[1,1].set_title('Component S', fontsize = 10)
plt.subplots_adjust(top = 0.9, bottom = 0.05)
figure8.savefig('ImgsReport/08_imgStraightLinesHSLGray')

'''
By showing the HSL components of the image, it can be seen that in the S component, the lines are very easy to see, without applying some
gradient, it seem like a good idea to create a binary image with this component
'''

# %% Binary image with HSL components

imgStrLin1UndistH_bin = binImg.HSLBinary(imgStraightLines1Undist, 'h', (180, 255))
imgStrLin1UndistS_bin = binImg.HSLBinary(imgStraightLines1Undist, 's', (180, 255))
imgStrLin1UndistL_bin = binImg.HSLBinary(imgStraightLines1Undist, 'l', (180, 255))

figure9, fig9_axes = plt.subplots(2,2, figsize = (10, 5))
figure9.tight_layout()
figure9.suptitle('Image "straight_lines1.jpg" and binary HSL components. Threshold = (150, 255)')
fig9_axes[0,0].imshow(imgStraightLines1Undist)
fig9_axes[0,0].set_title('Original image', fontsize = 10)
fig9_axes[0,1].imshow(imgStrLin1UndistH_bin, cmap = 'gray')
fig9_axes[0,1].set_title('Component H', fontsize = 10)
fig9_axes[1,0].imshow(imgStrLin1UndistL_bin, cmap = 'gray')
fig9_axes[1,0].set_title('Component L', fontsize = 10)
fig9_axes[1,1].imshow(imgStrLin1UndistS_bin, cmap = 'gray')
fig9_axes[1,1].set_title('Component S', fontsize = 10)
plt.subplots_adjust(top = 0.9, bottom = 0.05)
figure9.savefig('ImgsReport/09_imgStraightLinesHSLBin')

'''
By showing the binary images created thresholding the HSL components can be seen how good the lines are detected by the L and S components. 
The L component finds still more pieces of the line, but also another parts of the image, so it would not be smart to use it as it is without
applying gradient. Specially when the light is very strong, the L component could be very high everywhere, disturbing the line measurements
a lot. In the S component with the selected threshold almost only the lines can be seen, so it seems viable to use it to detect the lines.
'''

# %% Let's now identify the lines within all the other seven given test images

imgStraight_lines2 = mpimg.imread('test_images/straight_lines2.jpg')
imgTest1 = mpimg.imread('test_images/test1.jpg')
imgTest2 = mpimg.imread('test_images/test2.jpg')
imgTest3 = mpimg.imread('test_images/test3.jpg')
imgTest4 = mpimg.imread('test_images/test4.jpg')
imgTest5 = mpimg.imread('test_images/test5.jpg')
imgTest6 = mpimg.imread('test_images/test6.jpg')

# First they need to be undistorted
imgStraight_lines2ud = cam.UndistortImage(imgStraight_lines2, matrix, dist)
imgTest1ud = cam.UndistortImage(imgTest1, matrix, dist)
imgTest2ud = cam.UndistortImage(imgTest2, matrix, dist)
imgTest3ud = cam.UndistortImage(imgTest3, matrix, dist)
imgTest4ud = cam.UndistortImage(imgTest4, matrix, dist)
imgTest5ud = cam.UndistortImage(imgTest5, matrix, dist)
imgTest6ud = cam.UndistortImage(imgTest6, matrix, dist)

# Then their binary images created with the gradient in direction X using the component L
imgStraight_lines2_Lgrad = binImg.GradientCalc(imgStraight_lines2ud, imgType = 'HSL', imgChannel = 'l', calcType = 'dirX', thresh = (35, 190))
imgTest1_Lgrad = binImg.GradientCalc(imgTest1ud, imgType = 'HSL', imgChannel = 'l', calcType = 'dirX', thresh = (35, 160))
imgTest2_Lgrad = binImg.GradientCalc(imgTest2ud, imgType = 'HSL', imgChannel = 'l', calcType = 'dirX', thresh = (35, 160))
imgTest3_Lgrad = binImg.GradientCalc(imgTest3ud, imgType = 'HSL', imgChannel = 'l', calcType = 'dirX', thresh = (35, 160))
imgTest4_Lgrad = binImg.GradientCalc(imgTest4ud, imgType = 'HSL', imgChannel = 'l', calcType = 'dirX', thresh = (35, 160))
imgTest5_Lgrad = binImg.GradientCalc(imgTest5ud, imgType = 'HSL', imgChannel = 'l', calcType = 'dirX', thresh = (35, 160))
imgTest6_Lgrad = binImg.GradientCalc(imgTest6ud, imgType = 'HSL', imgChannel = 'l', calcType = 'dirX', thresh = (35, 160))

# And the binary images created with the S component
imgStraight_lines2_S = binImg.HSLBinary(imgStraight_lines2ud, 's', (180, 255))
imgTest1_S = binImg.HSLBinary(imgTest1ud, 's', (180, 250))
imgTest2_S = binImg.HSLBinary(imgTest2ud, 's', (180, 250))
imgTest3_S = binImg.HSLBinary(imgTest3ud, 's', (180, 250))
imgTest4_S = binImg.HSLBinary(imgTest4ud, 's', (180, 250))
imgTest5_S = binImg.HSLBinary(imgTest5ud, 's', (180, 250))
imgTest6_S = binImg.HSLBinary(imgTest6ud, 's', (180, 250))


# Finally, lets combine the images in bicolor images, in order to see which parts are selected by the gradient of the L component and which parts are
# selected by the S component
imgStraight_lines2_bin = binImg.CombineBinariesBlueGreen(imgStraight_lines2_Lgrad, imgStraight_lines2_S)
imgTest1_bin = binImg.CombineBinariesBlueGreen(imgTest1_Lgrad, imgTest1_S)
imgTest2_bin = binImg.CombineBinariesBlueGreen(imgTest2_Lgrad, imgTest2_S)
imgTest3_bin = binImg.CombineBinariesBlueGreen(imgTest3_Lgrad, imgTest3_S)
imgTest4_bin = binImg.CombineBinariesBlueGreen(imgTest4_Lgrad, imgTest4_S)
imgTest5_bin = binImg.CombineBinariesBlueGreen(imgTest5_Lgrad, imgTest5_S)
imgTest6_bin = binImg.CombineBinariesBlueGreen(imgTest6_Lgrad, imgTest6_S)

# And plot them
figure10, fig10_axes = plt.subplots(3,2, figsize = (11, 11))
figure10.tight_layout()
figure10.suptitle('Binary images with gradient in X direction of the component L in green and clean component S in blue. \n Threshold gradient L = (30, 150), Threshold S = (150, 255)')
fig10_axes[0,0].imshow(imgStraight_lines2)
fig10_axes[0,0].set_title('"straight_lines2.jpg"', fontsize = 10)
fig10_axes[0,0].axis('off')
fig10_axes[0,1].imshow(imgStraight_lines2_bin)
fig10_axes[0,1].set_title('Binary "straight_lines2.jpg"', fontsize = 10)
fig10_axes[0,1].axis('off')
fig10_axes[1,0].imshow(imgTest1)
fig10_axes[1,0].set_title('"test1.jpg"', fontsize = 10)
fig10_axes[1,0].axis('off')
fig10_axes[1,1].imshow(imgTest1_bin)
fig10_axes[1,1].set_title('Binary "test1.jpg"', fontsize = 10)
fig10_axes[1,1].axis('off')
fig10_axes[2,0].imshow(imgTest2)
fig10_axes[2,0].set_title('"test2.jpg"', fontsize = 10)
fig10_axes[2,0].axis('off')
fig10_axes[2,1].imshow(imgTest2_bin)
fig10_axes[2,1].set_title('Binary "test2.jpg"', fontsize = 10)
fig10_axes[2,1].axis('off')
plt.subplots_adjust(top = 0.9, bottom = 0.05)
figure10.savefig('ImgsReport/10_AllTestImagesBin01')

figure11, fig11_axes = plt.subplots(3,2, figsize = (11, 11))
figure11.tight_layout()
figure11.suptitle('Binary images with gradient in X direction of the component L in green and clean component S in blue. \n Threshold gradient L = (30, 150), Threshold S = (150, 255)')
fig11_axes[0,0].imshow(imgTest3)
fig11_axes[0,0].set_title('"test3.jpg"', fontsize = 10)
fig11_axes[0,0].axis('off')
fig11_axes[0,1].imshow(imgTest3_bin)
fig11_axes[0,1].set_title('Binary "test3.jpg"', fontsize = 10)
fig11_axes[0,1].axis('off')
fig11_axes[1,0].imshow(imgTest4)
fig11_axes[1,0].set_title('"test4.jpg"', fontsize = 10)
fig11_axes[1,0].axis('off')
fig11_axes[1,1].imshow(imgTest4_bin)
fig11_axes[1,1].set_title('Binary "test4.jpg"', fontsize = 10)
fig11_axes[1,1].axis('off')
fig11_axes[2,0].imshow(imgTest5)
fig11_axes[2,0].set_title('"test5.jpg"', fontsize = 10)
fig11_axes[2,0].axis('off')
fig11_axes[2,1].imshow(imgTest5_bin)
fig11_axes[2,1].set_title('Binary "test5.jpg"', fontsize = 10)
fig11_axes[2,1].axis('off')
plt.subplots_adjust(top = 0.9, bottom = 0.05)
figure11.savefig('ImgsReport/11_AllTestImagesBin02')


'''
By applying both methods on all images, can be seen that the combination of both is a very good approach to get the lane lines. What 
does not get detected by the gradient in direction X of the L component, gets detected with the S component and viceversa. 
This method will be used to get the lane line points and after that the equations corresponding to the lane lines.

A little optimization was here done with the parameters, the threshold range for the gradient in X was changed from (30, 150) to (35. 175)
and the threshold range for the S component was changed from (180, 255) to (180, 250) in order to prevent very strong shadows to appear 
in the binary image, like in "test5.jpg"
'''

# %%%%%%%%%%%%%%%%%%% PERSPECTIVE TRANSFORM %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

'''
Now that we have a robust method to get a binary image where the line pixels can be found, it is time to transform the images into
an aereal view, so the lines can be seen in a plane.
For that, the function WarpPolygonToSquare was added to the Camera class. It transforms the image using a polygon, so it receives four values
of X and two of Y.

Let's try that function with the first image "straight_lines1.jpg"
'''

imgStraightLines1Warped = cam.WarpPolygonToSquare(imgStraightLines1Undist, 450, 700, 200, 580, 1100, 700)

# Now, lets print it and compare it with the original image

figure12, fig12_axes = plt.subplots(1, 2, figsize = (10, 5))
figure12.tight_layout()
figure12.suptitle('Perspective transformed image')
fig12_axes[0].imshow(imgStraightLines1Undist)
fig12_axes[0].set_title('Original Image')
fig12_axes[1].imshow(imgStraightLines1Warped)
fig12_axes[1].set_title('Transformed image')
figure12.savefig('ImgsReport/12_StraightLines1_transformed')

'''
As it can be seen in the images, the perspective transform is working well getting an image in "bird view" perspective. The problem is
that the lines are not paralel as they should be. That is because the vertices of the selected polygon were selected very roughly. That the 
lines are paralel in the bird view image is very important in order to get good calculations after that. So it is needed to get the vertices
of that polygon in a precise way.

Here is where the work in the first project can be very useful to calculate those vertices. So we import from the first project functions 
"region_of_interest", "hough_vertices" and "hough_lines", now located in the file HoughLines.py and we update the "hough_lines" function to 
return the points at the beginning and at the end of the line.

Also the function "CombineBinaries" was added to the class BinaryImg. 
'''

# %%%%%%%%%%%%%%%%%%% PERSPECTIVE TRANSFORM WITH HOUGH LINES %%%%%%%%%%%%%%%%%%

exec(open('HoughLines.py').read())

# First, the binary image where the lane lines are visible needs to be made, to do that, the two images calculated before are used
imgStraightLines1bin =  binImg.CombineBinaries(strLines1BinHSLchLDirX, imgStrLin1UndistS_bin)

# Now the vertices where we calculate the Hough lines
y_horizon = 445
y_bottom = imgStraightLines1bin.shape[0] - 50

houghVertices = houghVertices(100, 550, 1200, 800, y_bottom, y_horizon)

# Then crop the image to the region of interest
houghRegion = region_of_interest(imgStraightLines1bin, houghVertices)

# Then calculate the hough lines, together with the points
houghLines, x_bottomLeft, x_topLeft, x_bottomRight, x_topRight = hough_lines(imgStraightLines1Undist, imgStraightLines1bin, 
                                                                                 1, np.pi/180, 50, 40, 2, y_horizon)
# Now with the new points the warped image corrected
imgStraightLines1Warped_corr = cam.WarpPolygonToSquare(imgStraightLines1Undist, y_horizon, imgStraightLines1Undist.shape[0], 
                                                       x_bottomLeft, x_topLeft, x_bottomRight, x_topRight)

# Lets draw the polygon in the original image
polygonVerticesOrig = np.array([[x_bottomLeft, y_bottom + 50],[x_topLeft, y_horizon],[x_topRight, y_horizon],[x_bottomRight, y_bottom + 50]], 
                           np.int32)
polygonVerticesWarped = np.array([[300, imgStraightLines1Warped_corr.shape[0]], [300, 0], [imgStraightLines1Warped_corr.shape[1] - 300, 0], 
                                  [imgStraightLines1Warped_corr.shape[1] - 300, imgStraightLines1Warped_corr.shape[0]]], np.int32)

imgStraightLines1Poly = imgStraightLines1Undist.copy()
imgStraightLines1Warped_corrPoly = imgStraightLines1Warped_corr.copy()
cv2.polylines(imgStraightLines1Poly, [polygonVerticesOrig], True, (255,0,0),3)
cv2.polylines(imgStraightLines1Warped_corrPoly, [polygonVerticesWarped], True, (255,0,0),4)

# Now let's print the entire process
figure13, fig13_axes = plt.subplots(2,2, figsize = (10, 5))
figure13.tight_layout()
figure13.suptitle('Perspective transformed image using polygon calculated with Hough lane lines')
fig13_axes[0,0].imshow(houghRegion, cmap = 'gray')
fig13_axes[0,0].set_title('Binary image masked with a region of interest', fontsize = 10)
fig13_axes[0,1].imshow(houghLines)
fig13_axes[0,1].set_title('Lane lines detected with Hough Lines Algorithm', fontsize = 10)
fig13_axes[1,0].imshow(imgStraightLines1Poly)
fig13_axes[1,0].set_title('Original image with transformation polygon', fontsize = 10)
fig13_axes[1,1].imshow(imgStraightLines1Warped_corrPoly)
fig13_axes[1,1].set_title('Transformated image', fontsize = 10)
plt.subplots_adjust(top = 0.9, bottom = 0.05)
figure13.savefig('ImgsReport/13_StraightLines1_transformed_corr')

'''
By using this method to find the vertices of the conversion polygon the lane lines in the converted image are completely paralel. What means
that the conversion is now accurate and appropiate to be used. The best thing is, that this process is not needed in the pipeline, these 
vertices can be used for any image to convert it into bird view image, so they can be used as parameters within the pipeline.
'''

# %% Lets validate the parameters by doing the conversion with other three images

imgTest1_warped = cam.WarpPolygonToSquare(imgTest1ud, y_horizon, imgTest1ud.shape[0], 
                                        x_bottomLeft, x_topLeft, x_bottomRight, x_topRight)

imgTest2_warped = cam.WarpPolygonToSquare(imgTest2ud, y_horizon, imgTest2ud.shape[0], 
                                        x_bottomLeft, x_topLeft, x_bottomRight, x_topRight)

imgTest3_warped = cam.WarpPolygonToSquare(imgTest3ud, y_horizon, imgTest3ud.shape[0], 
                                        x_bottomLeft, x_topLeft, x_bottomRight, x_topRight)

figure14, fig14_axes = plt.subplots(3,2, figsize = (11, 11))
figure14.tight_layout()
figure14.suptitle('Perspective transformed images using polygon calculated with Hough lane lines')
fig14_axes[0,0].imshow(imgTest1ud)
fig14_axes[0,0].set_title('"test1.jpg"', fontsize = 10)
fig14_axes[0,0].axis('off')
fig14_axes[0,1].imshow(imgTest1_warped)
fig14_axes[0,1].set_title('Warped "test1.jpg"', fontsize = 10)
fig14_axes[0,1].axis('off')
fig14_axes[1,0].imshow(imgTest2ud)
fig14_axes[1,0].set_title('"test2.jpg"', fontsize = 10)
fig14_axes[1,0].axis('off')
fig14_axes[1,1].imshow(imgTest2_warped)
fig14_axes[1,1].set_title('Warped "test2.jpg"', fontsize = 10)
fig14_axes[1,1].axis('off')
fig14_axes[2,0].imshow(imgTest3ud)
fig14_axes[2,0].set_title('"test3.jpg"', fontsize = 10)
fig14_axes[2,0].axis('off')
fig14_axes[2,1].imshow(imgTest3_warped)
fig14_axes[2,1].set_title('Warped "test5.jpg"', fontsize = 10)
fig14_axes[2,1].axis('off')
plt.subplots_adjust(top = 0.9, bottom = 0.05)
figure14.savefig('ImgsReport/14_TestImagesWarped')

'''
By showing the three images and their warped version it can be seen how well this polygon does performing the transformation. In all the three
images the lines stay paralel, which means that the conversion is valid and can be used for further analysis.
'''

# %%%%%%%%%%%%%%%%%%% BINARY IMAGE AND PERSPECTIVE TRANSFORM %%%%%%%%%%%%%%%%%%%%%%%%

'''
Now that the methods to get a binary image and to warp it into a bird view image are selected, programmed and tested with test images, it is 
important to decide: What to do first? get the binary image and then the perspective transform or viceversa?. For that, binary transformed 
images of some of the test images in both orders will be created to see which order is better.
'''

## First binary image, than perspective transform

imgTest1_bin = binImg.CombineBinaries(imgTest1_Lgrad, imgTest1_S)
imgTest2_bin = binImg.CombineBinaries(imgTest2_Lgrad, imgTest2_S)
imgTest3_bin = binImg.CombineBinaries(imgTest3_Lgrad, imgTest3_S)

imgTest1_binWarped = cam.WarpPolygonToSquare(imgTest1_bin, y_horizon, imgTest1ud.shape[0], 
                                        x_bottomLeft, x_topLeft, x_bottomRight, x_topRight)

imgTest2_binWarped = cam.WarpPolygonToSquare(imgTest2_bin, y_horizon, imgTest1ud.shape[0], 
                                        x_bottomLeft, x_topLeft, x_bottomRight, x_topRight)

imgTest3_binWarped = cam.WarpPolygonToSquare(imgTest3_bin, y_horizon, imgTest1ud.shape[0], 
                                        x_bottomLeft, x_topLeft, x_bottomRight, x_topRight)


## First perspective transform, than binary image

imgTest1_warpedLgrad = binImg.GradientCalc(imgTest1_warped, imgType = 'HSL', imgChannel = 'l', calcType = 'dirX', thresh = (35, 160))
imgTest2_warpedLgrad = binImg.GradientCalc(imgTest2_warped, imgType = 'HSL', imgChannel = 'l', calcType = 'dirX', thresh = (35, 160))
imgTest3_warpedLgrad = binImg.GradientCalc(imgTest3_warped, imgType = 'HSL', imgChannel = 'l', calcType = 'dirX', thresh = (35, 160))

imgTest1_warpedS = binImg.HSLBinary(imgTest1_warped, 's', (180, 250))
imgTest2_warpedS = binImg.HSLBinary(imgTest2_warped, 's', (180, 250))
imgTest3_warpedS = binImg.HSLBinary(imgTest3_warped, 's', (180, 250))

imgTest1_warpedBin = binImg.CombineBinaries(imgTest1_warpedLgrad, imgTest1_warpedS)
imgTest2_warpedBin = binImg.CombineBinaries(imgTest2_warpedLgrad, imgTest2_warpedS)
imgTest3_warpedBin = binImg.CombineBinaries(imgTest3_warpedLgrad, imgTest3_warpedS)

## Let's plot the first image
figure15, fig15_axes = plt.subplots(2,2, figsize = (10, 5))
figure15.tight_layout()
figure15.suptitle('Get binary image and then warp vs warp and then get binary image for "test2.jpg"')
fig15_axes[0,0].imshow(imgTest2ud)
fig15_axes[0,0].set_title('Original image', fontsize = 10)
fig15_axes[0,1].imshow(imgTest2_warped)
fig15_axes[0,1].set_title('Warped image', fontsize = 10)
fig15_axes[1,0].imshow(imgTest2_binWarped, cmap = 'gray')
fig15_axes[1,0].set_title('Binary and then warped', fontsize = 10)
fig15_axes[1,1].imshow(imgTest2_warpedBin, cmap = 'gray')
fig15_axes[1,1].set_title('Warped and then binary', fontsize = 10)
plt.subplots_adjust(top = 0.9, bottom = 0.05)
figure15.savefig('ImgsReport/15_ComparisonOrderTest2')

'''
For this image can be seen that both results are very similar, but in the warped image before getting the binary image with the lines
the right dashed line is not so well defined, also a little detail in the road is stronger detected in that version. Another thing is the 
car. While by creating first the binary image and then warping it the car does not get detected, but by warping the image and then creating
a binary image from it, the car gets detected, what can cause wrong measurements of the lines.

Let's plot the comparison with another image to be sure what option is the best
'''

# %% 

## Let's plot the first image
figure16, fig16_axes = plt.subplots(2,2, figsize = (10, 5))
figure16.tight_layout()
figure16.suptitle('Get binary image and then warp vs warp and then get binary image for "test1.jpg"')
fig16_axes[0,0].imshow(imgTest1ud)
fig16_axes[0,0].set_title('Original image', fontsize = 10)
fig16_axes[0,1].imshow(imgTest1_warped)
fig16_axes[0,1].set_title('Warped image', fontsize = 10)
fig16_axes[1,0].imshow(imgTest1_binWarped, cmap = 'gray')
fig16_axes[1,0].set_title('Binary and then warped', fontsize = 10)
fig16_axes[1,1].imshow(imgTest1_warpedBin, cmap = 'gray')
fig16_axes[1,1].set_title('Warped and then binary', fontsize = 10)
plt.subplots_adjust(top = 0.9, bottom = 0.05)
figure16.savefig('ImgsReport/16_ComparisonOrderTest1')


'''
In the case of this image, it can be seen that the left line gets detected longer in the version where first the binary image is created 
and then the binary image is warped than in the contrary version. Also in the contrary version some other details of the street are being
detected, which are not part of the line. So the better approach may be to get first the binary image and then warp it.
'''

# %%%%%%%%%%%%%%%%%%%%%%%%%% IDENTIFICATION OF LANE LINES ON WARPED BINARY IMAGE %%%%%%%%%%%%%%%%%

'''
Now it is time to calculate mathematically the lane lines by detecting their pixels on the warped binary images. For that a new class was programmed and
tested. The class "LinesProcessing" located in the file "ProcessLines.py", which contains the functions "findFirstLaneLinesPixels", 
"findNewLaneLinePixels" and "getLines".

The function "findFirstLaneLinesPixels" is meant to be used on the first frame of a video, or when the lines are completely lost. That is because it makes
some very heavy computation. In the normal case this function should only be used for the first frame of a video. 

In order to detect the lines for a new frame after the lines were detected for the frame before, the function "findNewLaneLinePixels" should be used.
This function uses the polynom coeficients of the lines of the last frame in order to get the lines of the current frame. It is way faster than the 
function "findFirstLaneLinesPixels" because it only contains one loop. An interesting option is to use the average of the polynomic coeficients of the lines
of the last n frames, wheiging more the ones of the last frames, getting so more robust calculations

To get the mathematical equations of the lane lines, the function "getLines" is used. This function interpolates the lane lines points found with
"findFirstLaneLinesPixels" or with "findNewLaneLinePixels". This function returns the polynomic coeficients of the two lines and the image with the calculated
line drawn on it.

Now let's try the functions "findFirstLaneLinesPixels" on the images "test1.jpg", "test2.jpg" and "test3.jpg". 
'''

# First, we need to create an object of type "LinesProcessing".
linesProc = LinesProcessing()
exec(open('ProcessLines.py').read())

# Then, the lane line pixels are calculated
imgTest1_LeftX, imgTest1_LeftY, imgTest1_RightX, imgTest1_RightY, imgTest1_LinePixels = linesProc.findFirstLaneLinesPixels(imgTest1_binWarped, showWindows = False,
                                                                                                                           paintLinePixels = True)
imgTest2_LeftX, imgTest2_LeftY, imgTest2_RightX, imgTest2_RightY, imgTest2_LinePixels = linesProc.findFirstLaneLinesPixels(imgTest2_binWarped, showWindows = False,
                                                                                                                           paintLinePixels = True)
imgTest3_LeftX, imgTest3_LeftY, imgTest3_RightX, imgTest3_RightY, imgTest3_LinePixels = linesProc.findFirstLaneLinesPixels(imgTest3_binWarped, showWindows = False,
                                                                                                                           paintLinePixels = True)

# Then the lines are calculated
imgTest1_leftCoeff, imgTest1_rightCoeff, imgTest1_Lines = linesProc.getLines(imgTest1_LinePixels,imgTest1_LeftX,imgTest1_LeftY,imgTest1_RightX,
                                                                            imgTest1_RightY, drawLines = True, drawLane = True)
imgTest2_leftCoeff, imgTest2_rightCoeff, imgTest2_Lines = linesProc.getLines(imgTest2_LinePixels,imgTest2_LeftX,imgTest2_LeftY,imgTest2_RightX,
                                                                            imgTest2_RightY, drawLines = True, drawLane = True)
imgTest3_leftCoeff, imgTest3_rightCoeff, imgTest3_Lines = linesProc.getLines(imgTest3_LinePixels,imgTest3_LeftX,imgTest3_LeftY,imgTest3_RightX,
                                                                            imgTest3_RightY, drawLines = True, drawLane = True)

figure17, fig17_axes = plt.subplots(3,1, figsize=(5,10))
figure17.tight_layout()
figure17.suptitle('Calculated lines on "test1.jpg", "test2.jpg" and "test3.jpg"')
fig17_axes[0].imshow(imgTest1_Lines)
fig17_axes[0].set_title('test1.jpg', fontsize = 10)
fig17_axes[1].imshow(imgTest2_Lines)
fig17_axes[1].set_title('test2.jpg', fontsize = 10)
fig17_axes[2].imshow(imgTest3_Lines)
fig17_axes[2].set_title('test3.jpg', fontsize = 10)
plt.subplots_adjust(top = 0.9, bottom = 0)
figure17.savefig('ImgsReport/17_CalculatedLines1')

'''
It can be seen that the functions work very well with these images, getting accurate curves. The only problem here seem to be that the lines are not 
completelly paralel. 

Now let's try the function "findNewLaneLinePixels" using an average of the polynomic coeficients calculated now.
'''

# %%%%%%%%%%%%%%%%%%%%%%%%%%% IDENTIFICATION OF LANE LINES ON WARPED BINARY IMAGE WITH FILTERING %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

## To do this, lets use the coeficients of the polynoms calculated on the last step, but considering that the coeficients of the second image weight more
## because these lines are more paralel

imgTest123_leftCoeff = (3*imgTest2_leftCoeff + imgTest1_leftCoeff + imgTest3_leftCoeff)/5
imgTest123_rightCoeff = (3*imgTest2_rightCoeff + imgTest1_rightCoeff + imgTest3_rightCoeff)/5

# Then, the lane line pixels are calculated
imgTest1_linePixels_return = linesProc.findNewLaneLinesPixels(imgTest1_binWarped, imgTest123_leftCoeff, imgTest123_rightCoeff, 
                                                             paintLinePixels = True, distCenter = 50)
imgTest1_LeftX2, imgTest1_LeftY2, imgTest1_RightX2, imgTest1_RightY2, imgTest1_LinePixels2 = imgTest1_linePixels_return
imgTest2_linePixels_return = linesProc.findNewLaneLinesPixels(imgTest2_binWarped, imgTest123_leftCoeff, imgTest123_rightCoeff, 
                                                             paintLinePixels = True, distCenter = 50)
imgTest2_LeftX2, imgTest2_LeftY2, imgTest2_RightX2, imgTest2_RightY2, imgTest2_LinePixels2 = imgTest2_linePixels_return
imgTest3_linePixels_return = linesProc.findNewLaneLinesPixels(imgTest3_binWarped, imgTest123_leftCoeff, imgTest123_rightCoeff, 
                                                             paintLinePixels = True, distCenter = 50)
imgTest3_LeftX2, imgTest3_LeftY2, imgTest3_RightX2, imgTest3_RightY2, imgTest3_LinePixels2 = imgTest3_linePixels_return

# Then the lines are calculated
imgTest1_leftCoeff2, imgTest1_rightCoeff2, imgTest1_Lines2 = linesProc.getLines(imgTest1_LinePixels2,imgTest1_LeftX2,imgTest1_LeftY2,imgTest1_RightX2,
                                                                            imgTest1_RightY2, drawLines = True, drawLane = True)
imgTest2_leftCoeff2, imgTest2_rightCoeff2, imgTest2_Lines2 = linesProc.getLines(imgTest2_LinePixels2,imgTest2_LeftX2,imgTest2_LeftY2,imgTest2_RightX2,
                                                                            imgTest2_RightY2, drawLines = True, drawLane = True)
imgTest3_leftCoeff2, imgTest3_rightCoeff2, imgTest3_Lines2 = linesProc.getLines(imgTest3_LinePixels2,imgTest3_LeftX2,imgTest3_LeftY2,imgTest3_RightX2,
                                                                            imgTest3_RightY2, drawLines = True, drawLane = True)

figure18, fig18_axes = plt.subplots(3,2, figsize=(7,7))
figure18.tight_layout()
figure18.suptitle('Calculated lines on "test1.jpg", "test2.jpg" and "test3.jpg"')
fig18_axes[0,0].imshow(imgTest1_Lines2)
fig18_axes[0,0].set_title('test1.jpg with filtering', fontsize = 10)
fig18_axes[0,1].imshow(imgTest1_Lines)
fig18_axes[0,1].set_title('test1.jpg without filtering', fontsize = 10)
fig18_axes[1,0].imshow(imgTest2_Lines2)
fig18_axes[1,0].set_title('test2.jpg with filtering', fontsize = 10)
fig18_axes[1,1].imshow(imgTest2_Lines)
fig18_axes[1,1].set_title('test2.jpg without filtering', fontsize = 10)
fig18_axes[2,0].imshow(imgTest3_Lines2)
fig18_axes[2,0].set_title('test3.jpg with filtering', fontsize = 10)
fig18_axes[2,1].imshow(imgTest3_Lines)
fig18_axes[2,1].set_title('test3.jpg without filtering', fontsize = 10)
plt.subplots_adjust(top = 0.9, bottom = 0)
figure18.savefig('ImgsReport/18_CalculatedLines2')


'''
Here it can be seen how the lines tend to be more similar to the other ones because of the ponderation. Specially in the case of "test3.jpg" it can be 
seen that the lines are inclined more to the left because of the influence of "test2.jpg" whose coeficients are of ponderation 3 of 5. Here gets evident
that a very good idea would be to use the coeficients of the last n lines, ponderating stronger the coeficients of the last frames, because they will be 
more similar to the current frame than the older coeficients.
'''

# %%%%%%%%%%%%%%%%%%%%%% CALCULATION OF THE RADIUS OF CURVATURE AND THE POSITION OF THE CAR %%%%%%%%%%%%%%%%%%%%%%%%%%

'''
To calculate the radius of curvature and the position of the car, three new functions were added to the class "LinesProcessing". These functions are
"getMeterPolynoms", "calculateCurvatureMeters" and "calculateVehiclePos". The function "getMeterPolynoms" does another interpolation in order to get
the polynomic coeficients in metric measurement units. These coeficients are used by the functions "calculateCurvatureMeters" and "calculateVehiclePos"
to calculate the radius of curvature of the lines and the position of the car relative to the center of the image.

Let's calculate these values for the images test1.jpg, test2.jpg and test3.jpg using the lines calculated without ponderation
'''

imgTest1_leftCoeffMetric, imgTest1_rightCoeffMetric = linesProc.getMeterPolynoms(imgTest1_LeftX,imgTest1_LeftY,imgTest1_RightX,imgTest1_RightY)
imgTest2_leftCoeffMetric, imgTest2_rightCoeffMetric = linesProc.getMeterPolynoms(imgTest2_LeftX,imgTest2_LeftY,imgTest2_RightX,imgTest2_RightY)
imgTest3_leftCoeffMetric, imgTest3_rightCoeffMetric = linesProc.getMeterPolynoms(imgTest3_LeftX,imgTest3_LeftY,imgTest3_RightX,imgTest3_RightY)

imgTest1_radLeft, imgTest1_radRight = linesProc.calculateCurvatureMeters(imgTest1_leftCoeffMetric, imgTest1_rightCoeffMetric)
imgTest2_radLeft, imgTest2_radRight = linesProc.calculateCurvatureMeters(imgTest2_leftCoeffMetric, imgTest2_rightCoeffMetric)
imgTest3_radLeft, imgTest3_radRight = linesProc.calculateCurvatureMeters(imgTest3_leftCoeffMetric, imgTest3_rightCoeffMetric)

imgTest1_posCar = linesProc.calculateVehiclePos(imgTest1_leftCoeffMetric, imgTest1_rightCoeffMetric)
imgTest2_posCar = linesProc.calculateVehiclePos(imgTest2_leftCoeffMetric, imgTest2_rightCoeffMetric)
imgTest3_posCar = linesProc.calculateVehiclePos(imgTest3_leftCoeffMetric, imgTest3_rightCoeffMetric)

print('Test1.jpg: Radius left: ', imgTest1_radLeft, ' [m] Radius right: ', imgTest1_radRight, ' [m] Car position: ', imgTest1_posCar, ' [m]')
print('Test2.jpg: Radius left: ', imgTest2_radLeft, ' [m] Radius right: ', imgTest2_radRight, ' [m] Car position: ', imgTest2_posCar, ' [m]')
print('Test3.jpg: Radius left: ', imgTest3_radLeft, ' [m] Radius right: ', imgTest3_radRight, ' [m] Car position: ', imgTest3_posCar, ' [m]')

# %%%%%%%%%%%%%%%%%%%%%%% UNWARP IMAGE WITH LANE LINES DETECTED %%%%%%%%%%%

'''
In order to draw the lane lines into the original undistorted image, two changes where made. The function "getLines" of the class "LinesProcessing" was 
extended in order to be able to draw a polygon, where the lane lines are located.
Also the function "UnwarpSquareToPolygon" was added to the class "Camera" in order to unwarp the image where the lane lines are drawn. This image will then
be added to the original unwarped image.

Let's execute this for the images "test1.jpg", "test2.jpg" and "test3.jpg"
'''

imgTest1_laneLinesUnwarped = cam.UnwarpSquareToPolygon(imgTest1_Lines, y_horizon, imgTest1ud.shape[0], 
                                                       x_bottomLeft, x_topLeft, x_bottomRight, x_topRight) 
imgTest2_laneLinesUnwarped = cam.UnwarpSquareToPolygon(imgTest2_Lines, y_horizon, imgTest1ud.shape[0], 
                                                       x_bottomLeft, x_topLeft, x_bottomRight, x_topRight) 
imgTest3_laneLinesUnwarped = cam.UnwarpSquareToPolygon(imgTest3_Lines, y_horizon, imgTest1ud.shape[0], 
                                                       x_bottomLeft, x_topLeft, x_bottomRight, x_topRight) 


figure19, fig19_axes = plt.subplots(3,1, figsize=(5,10))
figure19.tight_layout()
figure19.suptitle('Unwarped lane lines ready to be drawn into the original image')
fig19_axes[0].imshow(imgTest1_laneLinesUnwarped)
fig19_axes[0].set_title('test1.jpg', fontsize = 10)
fig19_axes[1].imshow(imgTest2_laneLinesUnwarped)
fig19_axes[1].set_title('test2.jpg', fontsize = 10)
fig19_axes[2].imshow(imgTest3_laneLinesUnwarped)
fig19_axes[2].set_title('test3.jpg', fontsize = 10)
plt.subplots_adjust(top = 0.9, bottom = 0)
figure19.savefig('ImgsReport/19_UnwarpedLines')

'''
It can be seen that it works pretty well for the three images.

Now, the last part of the pipeline. Combine this images with the original, including the radius of curvature and the position of the vehicle
'''

# %%%%%%%%%%%%%%%% ADD UNWARPED IMAGE WITH LANE ON IT TO THE ORIGINAL UNDISTORTED IMAGE %%%%%%%%%%%%%%%%%%

'''
In order to get the final image with the lane lines on it and the calculations, the function "addDataToOriginal" was added to the class "LinesProcessing".
Let's try it with the 3 test images.
'''

# Only to update file
exec(open('ProcessLines.py').read())
linesProc = LinesProcessing()

imgTest1out = linesProc.addDataToOriginal(imgTest1ud, imgTest1_laneLinesUnwarped, imgTest1_radLeft, imgTest1_radRight, imgTest1_posCar)
imgTest2out = linesProc.addDataToOriginal(imgTest2ud, imgTest2_laneLinesUnwarped, imgTest2_radLeft, imgTest2_radRight, imgTest2_posCar)
imgTest3out = linesProc.addDataToOriginal(imgTest3ud, imgTest3_laneLinesUnwarped, imgTest3_radLeft, imgTest3_radRight, imgTest3_posCar)

figure20, fig20_axes = plt.subplots(3,1, figsize=(5,10))
figure20.tight_layout()
figure20.suptitle('Output images')
fig20_axes[0].imshow(imgTest1out)
fig20_axes[0].set_title('test1.jpg', fontsize = 10)
fig20_axes[1].imshow(imgTest2out)
fig20_axes[1].set_title('test2.jpg', fontsize = 10)
fig20_axes[2].imshow(imgTest3out)
fig20_axes[2].set_title('test3.jpg', fontsize = 10)
plt.subplots_adjust(top = 0.9, bottom = 0)
plt.imsave("output_images/test1_output.jpg", imgTest1out, format = 'jpg')
plt.imsave("output_images/test2_output.jpg", imgTest2out, format = 'jpg')
plt.imsave("output_images/test3_output.jpg", imgTest3out, format = 'jpg')
figure20.savefig('ImgsReport/20_OutputImages')


'''
It can be seen that the lane lines are good identified in the three output images, indicating that the pipeline is ready and working well.
'''

# %%%%%%%%%%%%%%%%%%%%%%%% VALIDATION OF THE PIPELINE %%%%%%%%%%%%%%%%%%%%%%%%%

'''
Now let's recapitulate what are the steps of the pipeline:
    1) Undistort the image using the function "UndistortImage" of "Camera" and the camera matrix and distortion coeficients obtained with 
    the camera calibration.
    2) Obtain a binary image of the undistorted image by calculating the gradient in direction X of its L color channel using the function 
    "GradientCalc" of the class "BinaryImg". Default threshold range is (35,180)
    3) Obtain a binary image of the undistorted image by thresholding its S color channel using the function "HSLBinary" of the class "BinaryImg".
    Default threshold range is (180, 250)
    4) Combine both binary images with the function "CombineBinaries" of the class "BinaryImg"
    5) Warp the combined binary image into a "bird view" image with the function "WarpPolygonToSquare" of the class "Camera" using the polygon parameters
    calculated with the function "hough_lines" created for the first project.
    6) Get the lane lines pixels from the warped binary image using the function "findFirstLaneLinesPixels" of the class "LinesProcessing" if the first 
    frame or an image is being processed. Use the function "findNewLaneLinesPixels" instead, if the lines were identified already on another frame 
    or frames. If wanted, paints the left line pixels red and the right line pixels blue.
    7) Get the coeficients of second grade polynoms which defines the lines with the function "getLines" of the class "LinesProcessing". If wanted draws
    the lane lines and/or the lane into the image.
    8) Get the coeficients of second grade polynoms which defines the lines in real measurement units with the function "getMeterPolynoms" of the class
    "LinesProcessing" to be used in order to calculate the radius of curvature of the lines and the position of the car relative to the center.
    9) Calculate the radius of curvature of both lines using the function "calculateCurvatureMeters" of the class "LinesProcessing". 
    10) Calculate the position of the car relative to the center of the image using the function "calculateVehiclePos" of the class "LinesProcessing".
    11) Unwarp the image with the lane lines and the line drawn on it with the function "UnwarpSquareToPolygon" of the class "Camera" using
    the same parameters used to warp the image on the step 5.
    12) Overlap the unwarped image into the original undistorted image using the function "addDataToOriginal" of the class "LinesProcessing". It also
    prints the radius of curvature of both lines and the position of the car into the output image.
    
    Let's define here a function which does all the steps.
'''


# exec(open('ProcessVideo.py').read())
# Let's try the pipeline with the images test4.jpg, test5.jpg and test6.jpg

def getStartLaneLines(image, camMatrix, distCoeff, warpParameters, gradXLThresh = (35, 180), SThresh = (180, 250)):
    # Undistort the image using the function "UndistortImage" of "Camera" and the camera matrix and distortion coeficients obtained with 
    # the camera calibration.
    undistImg = cam.UndistortImage(image, camMatrix, distCoeff)
    # Obtain a binary image of the undistorted image by calculating the gradient in direction X of its L color channel using the function 
    # "GradientCalc" of the class "BinaryImg".
    binImgGradX_Lch = binImg.GradientCalc(undistImg, imgType = 'HSL', imgChannel = 'l', calcType = 'dirX', kernelSize = 3, thresh = gradXLThresh)
    # Obtain a binary image of the undistorted image by thresholding its S color channel using the function "HSLBinary" of the class "BinaryImg".
    binImg_Sch = binImg.HSLBinary(undistImg, imgChannel = 's', thresh = SThresh)
    # Combine both binary images with the function "CombineBinaries" of the class "BinaryImg"
    binImage = binImg.CombineBinaries(binImgGradX_Lch, binImg_Sch)
    # Warp the combined binary image into a "bird view" image with the function "WarpPolygonToSquare" of the class "Camera" using the polygon parameters
    # calculated with the function "hough_lines" created for the first project.
    binWarpedImage = cam.WarpPolygonToSquare(binImage, warpParameters[0], warpParameters[1], warpParameters[2], 
                                             warpParameters[3], warpParameters[4], warpParameters[5])
    # Get the lane lines pixels from the warped binary image using the function "findFirstLaneLinesPixels" of the class "LinesProcessing". Paints the left
    # line pixels blue and the right line pixels red
    LeftX, LeftY, RightX, RightY, pixelsBinWarpedImage = linesProc.findFirstLaneLinesPixels(binWarpedImage, showWindows = False, paintLinePixels = True)
    # Get the coeficients of second grade polynoms which defines the lines with the function "getLines" of the class "LinesProcessing". If wanted draws
    # the lane lines and/or the lane into the image.
    leftCoef, rightCoef, laneBinWarpedImage = linesProc.getLines(pixelsBinWarpedImage,LeftX,LeftY,RightX,RightY, drawLines = True, drawLane = True)
    # Get the coeficients of second grade polynoms which defines the lines in real measurement units with the function "getMeterPolynoms" of the class
    # "LinesProcessing" to be used in order to calculate the radius of curvature of the lines and the position of the car relative to the center.
    leftCoefMetric, rightCoefMetric = linesProc.getMeterPolynoms(LeftX,LeftY,RightX,RightY)
    # Calculate the radius of curvature of both lines using the function "calculateCurvatureMeters" of the class "LinesProcessing". 
    radLeft, radRight = linesProc.calculateCurvatureMeters(leftCoefMetric, rightCoefMetric)
    # Calculate the position of the car relative to the center of the image using the function "calculateVehiclePos" of the class "LinesProcessing".
    posCar = linesProc.calculateVehiclePos(leftCoefMetric, rightCoefMetric)
    # Unwarp the image with the lane lines and the line drawn on it with the function "UnwarpSquareToPolygon" of the class "Camera" using
    # the same parameters used to warp the image on the step 5.
    laneUnwarpedImage = cam.UnwarpSquareToPolygon(laneBinWarpedImage, warpParameters[0], warpParameters[1], warpParameters[2], 
                                                  warpParameters[3], warpParameters[4], warpParameters[5])
    # Overlap the unwarped image into the original undistorted image using the function "addDataToOriginal" of the class "LinesProcessing". It also
    # prints the radius of curvature of both lines and the position of the car into the output image.
    imageOutput = linesProc.addDataToOriginal(undistImg, laneUnwarpedImage, radLeft, radRight, posCar)

wParameters = [y_horizon, 720, x_bottomLeft, x_topLeft, x_bottomRight, x_topRight]

imgTest4out, leftCoef4, rightCoef4, radLeft4, radRight4, posCar4 = getStartLaneLines(imgTest4, matrix, dist, wParameters, gradXLThresh = (35, 180), SThresh = (180, 250))
imgTest5out, leftCoef5, rightCoef5, radLeft5, radRight5, posCar5 = getStartLaneLines(imgTest5, matrix, dist, wParameters, gradXLThresh = (35, 180), SThresh = (180, 250))
imgTest6out, leftCoef6, rightCoef6, radLeft6, radRight6, posCar = getStartLaneLines(imgTest6, matrix, dist, wParameters, gradXLThresh = (35, 180), SThresh = (180, 250))

figure21, fig21_axes = plt.subplots(3,1, figsize=(5,10))
figure21.tight_layout()
figure21.suptitle('Output images. Generalized pipeline')
fig21_axes[0].imshow(imgTest4out)
fig21_axes[0].set_title('test4.jpg', fontsize = 10)
fig21_axes[1].imshow(imgTest5out)
fig21_axes[1].set_title('test5.jpg', fontsize = 10)
fig21_axes[2].imshow(imgTest6out)
fig21_axes[2].set_title('test6.jpg', fontsize = 10)
plt.subplots_adjust(top = 0.9, bottom = 0)
plt.imsave("output_images/test4_output.jpg", imgTest4out, format = 'jpg')
plt.imsave("output_images/test5_output.jpg", imgTest5out, format = 'jpg')
plt.imsave("output_images/test6_output.jpg", imgTest6out, format = 'jpg')
figure21.savefig('ImgsReport/21_OutputImages2')



# %%% FRAME TO FRAME CHALLENGE VIDEO %%%%

'''
For the challenge video, the lines are not getting detected as they should because the street has some other elements which resemble lines. In order
to get this better, the process of getting the binary image needs to be revised.
Let's do all the pipeline for two frames of that video to see where it fails:
'''

from moviepy.editor import VideoFileClip
clipChallenge = VideoFileClip('challenge_video.mp4')

challengeFr1 = clipChallenge.get_frame(1)
clipChallenge.save_frame('test_images/challengeFr1.jpg', t=1)
challengeFr2 = clipChallenge.get_frame(4)
clipChallenge.save_frame('test_images/challengeFr2.jpg', t=4)

# Undistort
challengeFr1ud = cam.UndistortImage(challengeFr1, matrix, dist)
challengeFr2ud = cam.UndistortImage(challengeFr2, matrix, dist)

# Get binary images
challengeFr1GradX_Lch = binImg.GradientCalc(challengeFr1ud, imgType = 'HSL', imgChannel = 'l', calcType = 'dirX', kernelSize = 3, thresh = (35, 180))
challengeFr1S_ch = binImg.HSLBinary(challengeFr1ud, imgChannel = 's', thresh = (180, 250))
challengeFr2GradX_Lch = binImg.GradientCalc(challengeFr2ud, imgType = 'HSL', imgChannel = 'l', calcType = 'dirX', kernelSize = 3, thresh = (35, 180))
challengeFr2S_ch = binImg.HSLBinary(challengeFr2ud, imgChannel = 's', thresh = (180, 250))

challengeFr1binCombined = binImg.CombineBinariesBlueGreen(challengeFr1GradX_Lch, challengeFr1S_ch)
challengeFr2binCombined = binImg.CombineBinariesBlueGreen(challengeFr2GradX_Lch, challengeFr2S_ch)

figure22, fig22_axes = plt.subplots(2,2, figsize = (10, 5))
figure22.tight_layout()
figure22.suptitle('Binary images of two frames of "challenge_video.mp4"')
fig22_axes[0,0].imshow(challengeFr1)
fig22_axes[0,0].set_title('Frame at 1 [s]', fontsize = 10)
fig22_axes[0,1].imshow(challengeFr1binCombined)
fig22_axes[0,1].set_title('Binary image at 1 [s]', fontsize = 10)
fig22_axes[1,0].imshow(challengeFr2)
fig22_axes[1,0].set_title('Frame at 4 [s]', fontsize = 10)
fig22_axes[1,1].imshow(challengeFr2binCombined)
fig22_axes[1,1].set_title('Binary image at 4 [s]', fontsize = 10)
plt.subplots_adjust(top = 0.9, bottom = 0.05)
figure22.savefig('ImgsReport/22_BinaryImages2Frames')

''' 
It can be seen that the gradient on the X direction of the channel L detects some lines, but the channel S does not detect anything here. Also the lines
of the gradient are incorrect. Lets decrease the threshold range of the gradient and increase the one of the S channel
'''

# %%

# Get binary images
challengeFr1GradX_Lch2 = binImg.GradientCalc(challengeFr1ud, imgType = 'HSL', imgChannel = 'l', calcType = 'dirX', kernelSize = 3, thresh = (30, 180))
challengeFr1L_ch2 = binImg.HSLBinary(challengeFr1ud, imgChannel = 'l', thresh = (170, 255))
challengeFr1S_ch2 = binImg.HSLBinary(challengeFr1ud, imgChannel = 's', thresh = (120, 250))

challengeFr1GradX_Lch2_L = cv2.bitwise_and(challengeFr1GradX_Lch2, challengeFr1L_ch2)

challengeFr2GradX_Lch2 = binImg.GradientCalc(challengeFr2ud, imgType = 'HSL', imgChannel = 'l', calcType = 'dirX', kernelSize = 3, thresh = (30, 180))
challengeFr2L_ch2 = binImg.HSLBinary(challengeFr2ud, imgChannel = 'l', thresh = (170, 255))
challengeFr2S_ch2 = binImg.HSLBinary(challengeFr2ud, imgChannel = 's', thresh = (120, 250))

challengeFr2GradX_Lch2_L = cv2.bitwise_and(challengeFr2GradX_Lch2, challengeFr2L_ch2)

challengeFr1binCombined2 = binImg.CombineBinariesBlueGreen(challengeFr1GradX_Lch2_L, challengeFr1S_ch2)
challengeFr2binCombined2 = binImg.CombineBinariesBlueGreen(challengeFr2GradX_Lch2_L, challengeFr2S_ch2)

figure23, fig23_axes = plt.subplots(2,2, figsize = (10, 5))
figure23.tight_layout()
figure23.suptitle('Binary images of two frames of "challenge_video.mp4" \n with parameters changed.')
fig23_axes[0,0].imshow(challengeFr1)
fig23_axes[0,0].set_title('Frame at 1 [s]', fontsize = 10)
fig23_axes[0,1].imshow(challengeFr1binCombined2)
fig23_axes[0,1].set_title('Binary image at 1 [s]', fontsize = 10)
fig23_axes[1,0].imshow(challengeFr2)
fig23_axes[1,0].set_title('Frame at 4 [s]', fontsize = 10)
fig23_axes[1,1].imshow(challengeFr2binCombined2)
fig23_axes[1,1].set_title('Binary image at 4 [s]', fontsize = 10)
plt.subplots_adjust(top = 0.9, bottom = 0.05)
figure23.savefig('ImgsReport/23_BinaryImages2FramesImproved')

'''
After a lot of test, I found out that the dark lines which are not street lines can be filtered from the gradient in the direction X by masking it only 
taking the pixels with a high l value (180 to 255). Also the threshold range of the gradient in direction X of the L channel was increased to (10, 230) and 
the threshold range of the S channel was increased to (100, 250)
'''

