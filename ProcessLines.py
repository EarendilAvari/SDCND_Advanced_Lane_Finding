#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  8 19:22:06 2019

@author: earendilavari
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

class LinesProcessing:
    def findLaneLinesPixels (self, image, heighWindows = 80, distCenter = 75, minPix = 50, drawWindows = True, paintLinePixels = True):
        # This function is similar to "find_lane_pixels" of the lesson "Advanced computer vision"
        # it gets the X and Y values of the pixels corresponding to the right and left lane lines.
        # It uses the slicing window method to diferentiate which pixels correspond to the left 
        # and right lane line.
        
        # Creates a 3-channel image to visualize the results and to be used further
        imgPixels = np.dstack((image, image, image))
        
        # Gets the histogram of the bottom half of the image 
        hist = np.sum(image[image.shape[0]//2:,:], axis=0)
        
        # Finds the column with the most activated pixels for the right and left lines to be
        # used as start points
        imgMidPoint = np.int(hist.shape[0]//2)
        leftStartMidPoint = np.argmax(hist[:imgMidPoint])
        rightStartMidPoint = np.argmax(hist[imgMidPoint:]) + imgMidPoint
        
        # Determines the quantity of windows based of its heigh
        nWindows = np.int(image.shape[0]//heighWindows)
        
        # Identifies the non zero pixels on the image. This values correspond
        # to the indexes where the nonzero pixels are located, so a non zero point
        # can be found like (nonZeroX[i], nonZeroY[i])
        nonZero = image.nonzero()
        nonZeroX = np.array(nonZero[1])
        nonZeroY = np.array(nonZero[0])
        
        # Center position of the window which is currently being processed, for the first
        # image they correspond to the start points defined before through the histogram
        leftCurrMidPoint = leftStartMidPoint
        rightCurrMidPoint = rightStartMidPoint
        
        # These two lists will contain the indices on nonZeroX and nonZeroY that correspond
        # to pixels in the left line and in the right line
        leftLineInds = []
        rightLineInds = []
        
        # Iterates through every level of windows
        
        for i in range(nWindows):
            # Define the values that limit the windows
            winY_low = image.shape[0] - (i+1)*heighWindows
            winY_high = image.shape[0] - i*heighWindows
            winX_left_low = leftCurrMidPoint - distCenter 
            winX_left_high = leftCurrMidPoint + distCenter 
            winX_right_low = rightCurrMidPoint - distCenter  
            winX_right_high = rightCurrMidPoint + distCenter
        
            # Draws the windows on the output image
            if drawWindows == 'True':
                cv2.rectangle(imgPixels,(winX_left_low,winY_low),(winX_left_high,winY_high),(0,255,0), 2) 
                cv2.rectangle(imgPixels,(winX_right_low,winY_low),(winX_right_high,winY_high),(0,255,0), 2) 
        
            currPixInLeftWindow = 0     #This number needs to be higher than minPix to update the X position of the window
            currPixInRightWindow = 0    #This number needs to be higher than minPix to update the X position of the window
            
            for j in range (0,nonZeroX.shape[0] - 1):
                if ((nonZeroX[j] > winX_left_low) & (nonZeroX[j] < winX_left_high) & 
                (nonZeroY[j] >= winY_low) & (nonZeroY[j] < winY_high)):
                    leftLineInds.append(j)
                    currPixInLeftWindow += 1
                if ((nonZeroX[j] > winX_right_low) & (nonZeroX[j] < winX_right_high) & 
                (nonZeroY[j] >= winY_low) & (nonZeroY[j] < winY_high)):
                    rightLineInds.append(j)
                    currPixInRightWindow += 1       
                
            # The centers of the windows are updated if there is enough points inside of them
            if (currPixInLeftWindow > minPix):
                histLeftWindow = np.sum(image[winY_low:winY_high,winX_left_low:winX_left_high], axis = 0)
                leftCurrMidPoint = leftCurrMidPoint - distCenter + np.argmax(histLeftWindow)
        
            if (currPixInRightWindow > minPix):
                histRightWindow = np.sum(image[winY_low:winY_high,winX_right_low:winX_right_high], axis = 0)
                rightCurrMidPoint = rightCurrMidPoint - distCenter + np.argmax(histRightWindow)
    
        # Extract left and right line pixel positions
        leftX = nonZeroX[leftLineInds]
        leftY = nonZeroY[leftLineInds] 
        rightX = nonZeroX[rightLineInds]
        rightY = nonZeroY[rightLineInds]
        
        # Paints the pixels on the left line blue and in the right line red
        if paintLinePixels == 'True':
            imgPixels[leftY, leftX] = [255,0,0]
            imgPixels[rightY, rightX] = [0, 0, 255]
        
        
        return leftX, leftY, rightX, rightY, imgPixels
    
    def getLines(self, image, leftX, leftY, rightX, rightY, drawLines = 'True'):
        # Gets the coeficients of a second grade polynom defining the right and the left lines.
        # If desired, it draws the lines into the image
        
        leftLineCoeficients = np.polyfit(leftY, leftX, 2)
        rightLineCoeficients = np.polyfit(rightY, rightX, 2)
        
        # Creates a copy of the input image where the lines will be drawn
        imageLines = image.copy()
        
        linesY = np.linspace(0, image.shape[0]-1, image.shape[0])
        
        try:
            leftLineX = leftLineCoeficients[0]*linesY**2 + leftLineCoeficients[1]*linesY + leftLineCoeficients[2]
            rightLineX = rightLineCoeficients[0]*linesY**2 + rightLineCoeficients[1]*linesY + rightLineCoeficients[2]
        except TypeError:
            # Avoids an error if `left` and `right_fit` are still none or incorrect
            print('The function failed to fit a line!')
            leftLineX = 1*linesY**2 + 1*linesY
            rightLineX = 1*linesY**2 + 1*linesY

        if drawLines == 'True':
            imageLines[np.int_(linesY), np.int_(leftLineX), 0] = 255
            imageLines[np.int_(linesY), np.int_(leftLineX), 1] = 255
            imageLines[np.int_(linesY), np.int_(leftLineX) + 1, 0] = 255
            imageLines[np.int_(linesY), np.int_(leftLineX) + 1, 1] = 255
            imageLines[np.int_(linesY), np.int_(leftLineX) - 1, 0] = 255
            imageLines[np.int_(linesY), np.int_(leftLineX) - 1, 1] = 255
            imageLines[np.int_(linesY), np.int_(rightLineX), 0] = 255
            imageLines[np.int_(linesY), np.int_(rightLineX), 1] = 255
            imageLines[np.int_(linesY), np.int_(rightLineX) + 1, 0] = 255
            imageLines[np.int_(linesY), np.int_(rightLineX) + 1, 1] = 255
            imageLines[np.int_(linesY), np.int_(rightLineX) - 1, 0] = 255
            imageLines[np.int_(linesY), np.int_(rightLineX) - 1, 1] = 255
            
        return leftLineCoeficients, rightLineCoeficients, imageLines
    

# Load our image
binary_warped = mpimg.imread('warped_example.jpg')

linesProc = LinesProcessing()

leftX, leftY, rightX, rightY, test1 = linesProc.findLaneLinesPixels(binary_warped, drawWindows = 'True', paintLinePixels = 'True' )
leftLineCoeff, rightLineCoeff, test2 = linesProc.getLines(test1, leftX, leftY, rightX, rightY, drawLines = 'True')

plt.imshow(test2)