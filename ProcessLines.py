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
    '''
    def __init__(self):
        self.lastLeftLineCoeficients = None
        self.lastRightLineCoeficients = None
        self.lastLeftLineCoeficientsMeters = None
        self.lastRightLineCoeficientsMeters = None
    '''
    def findFirstLaneLinesPixels(self, image, heighWindows = 80, distCenter = 75, minPix = 50, showWindows = True, paintLinePixels = True):
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
            if showWindows == True:
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
                if len(histLeftWindow) > 0:
                    leftCurrMidPoint = leftCurrMidPoint - distCenter + np.argmax(histLeftWindow)
        
            if (currPixInRightWindow > minPix):
                histRightWindow = np.sum(image[winY_low:winY_high,winX_right_low:winX_right_high], axis = 0)
                if len(histRightWindow) > 0:
                    rightCurrMidPoint = rightCurrMidPoint - distCenter + np.argmax(histRightWindow)
    
        # Extract left and right line pixel positions
        leftX = nonZeroX[leftLineInds]
        leftY = nonZeroY[leftLineInds] 
        rightX = nonZeroX[rightLineInds]
        rightY = nonZeroY[rightLineInds]
        
        # Paints the pixels on the left line blue and in the right line red
        if paintLinePixels == True:
            imgPixels[leftY, leftX] = [255,0,0]
            imgPixels[rightY, rightX] = [0, 0, 255]
        
        return leftX, leftY, rightX, rightY, imgPixels
    
    def findNewLaneLinesPixels(self, image, coeffLastLeftLine, coeffLastRightLine, distCenter = 75, paintLinePixels = True):
        # This function is similar to "findFirstLaneLinesPixels", but it finds the new pixel values 
        # based on the last found lines, or better said, based on their polynom coeficients.
        # It gets the X and Y values of the pixels corresponding to the right and left lane lines.
        # It searches for the pixels close to the last found lines in order to optimize the search.
        
        # Creates a 3-channel image to visualize the results and to be used further
        imgPixels = np.dstack((image, image, image))
        
        # Identifies the non zero pixels on the image. This values correspond
        # to the indexes where the nonzero pixels are located, so a non zero point
        # can be found like (nonZeroX[i], nonZeroY[i])
        nonZero = image.nonzero()
        nonZeroX = np.array(nonZero[1])
        nonZeroY = np.array(nonZero[0])
        
        # These two lists will contain the indices on nonZeroX and nonZeroY that correspond
        # to pixels in the left line and in the right line
        leftLineInds = []
        rightLineInds = []
        
        for i in range(0, nonZeroY.shape[0] - 1):
            xLeftCurr = coeffLastLeftLine[0]*nonZeroY[i]**2 + coeffLastLeftLine[1]*nonZeroY[i] + coeffLastLeftLine[2]
            xRightCurr = coeffLastRightLine[0]*nonZeroY[i]**2 + coeffLastRightLine[1]*nonZeroY[i] + coeffLastRightLine[2]
            if ((nonZeroX[i] > (xLeftCurr - distCenter)) & (nonZeroX[i] < (xLeftCurr + distCenter))):
                leftLineInds.append(i)
            if ((nonZeroX[i] > (xRightCurr - distCenter)) & (nonZeroX[i] < (xRightCurr + distCenter))):
                rightLineInds.append(i)
                
        # Extract left and right line pixel positions
        leftX = nonZeroX[leftLineInds]
        leftY = nonZeroY[leftLineInds] 
        rightX = nonZeroX[rightLineInds]
        rightY = nonZeroY[rightLineInds]
        
        # Paints the pixels on the left line blue and in the right line red
        if paintLinePixels == True:
            imgPixels[leftY, leftX] = [255,0,0]
            imgPixels[rightY, rightX] = [0, 0, 255]

        return leftX, leftY, rightX, rightY, imgPixels            
        
    
    def getLines(self, image, leftX, leftY, rightX, rightY, drawLines = True, drawLane = True):
        # Gets the coeficients of a second grade polynom defining the right and the left lines.
        # If desired, it draws the lines into the image
        
        try: 
            leftLineCoeficients = np.polyfit(leftY, leftX, 2)
            rightLineCoeficients = np.polyfit(rightY, rightX, 2)
        except TypeError:
            '''
            leftLineCoeficients = self.lastLeftLineCoeficients
            rightLineCoeficients = self.lastRightLineCoeficients
            '''
            leftLineCoeficients = np.array([0,0,0])
            rightLineCoeficients = np.array([0,0,0])
        
        # Creates a copy of the input image where the lines will be drawn
        imageLines = image.copy()
        
        linesY = np.linspace(0, image.shape[0]-1, image.shape[0])
        

        leftLineX = leftLineCoeficients[0]*linesY**2 + leftLineCoeficients[1]*linesY + leftLineCoeficients[2]
        rightLineX = rightLineCoeficients[0]*linesY**2 + rightLineCoeficients[1]*linesY + rightLineCoeficients[2]
        
        '''
        self.lastLeftLineCoeficients = leftLineCoeficients
        self.lastRightLineCoeficients = rightLineCoeficients
        '''

        if drawLines == True:
            for i in range(0, image.shape[0] - 1):
                if ((leftLineX[i] > 0) & (leftLineX[i] < image.shape[1] - 1)): 
                    imageLines[np.int_(linesY[i]), np.int_(leftLineX[i]), 1:] = 255
                    imageLines[np.int_(linesY[i]), np.int_(leftLineX[i]) + 1, 1:] = 255
                    imageLines[np.int_(linesY[i]), np.int_(leftLineX[i]) - 1, 1:] = 255
                if ((rightLineX[i] > 0) & (rightLineX[i] < image.shape[1] - 1)): 
                    imageLines[np.int_(linesY[i]), np.int_(rightLineX[i]), 1:] = 255
                    imageLines[np.int_(linesY[i]), np.int_(rightLineX[i]) + 1, 1:] = 255
                    imageLines[np.int_(linesY[i]), np.int_(rightLineX[i]) - 1, 1:] = 255
        
        # If "drawLane" is true, the lane line gets drawn into the image. Feature which is useful when the image gets
        # unwarped and overlaped to the original undistorted image
        if drawLane == True:
            # Creates an empty image where the lane will be drawn
            imageLane = np.zeros_like(imageLines)
            # Generates a polygon indicating where the lane is located
            # leftLineBorder is an array of all the points (x,y) which belong to the left line organized upwards
            leftLineBorder = np.array([np.transpose(np.vstack([leftLineX, linesY]))])
            # rightLineBorder is an array of all the points (x,y) which belong to the right line, but this time organized downwards
            rightLineBorder = np.array([np.flipud(np.transpose(np.vstack([rightLineX, linesY])))])
            # lanePoints is an array of all the points which define the polygon drawing the lane
            lanePoints = np.hstack((leftLineBorder, rightLineBorder))
            
            # Draws the lane into the blank image created before
            cv2.fillPoly(imageLane, np.int_([lanePoints]), (0,255,0))
            # Adds the image to the one meant to be returned
            imageLaneAndLines = cv2.addWeighted(imageLines, 1, imageLane, 0.5, 0)
        else:
            imageLaneAndLines = imageLines
            
        return leftLineCoeficients, rightLineCoeficients, imageLaneAndLines
    
    def getPolynoms(self, leftX, leftY, rightX, rightY):
        # Gets the coeficients of a second grade polynom defining the right and the left lines.
        
        try: 
            leftLineCoeficients = np.polyfit(leftY, leftX, 2)
            rightLineCoeficients = np.polyfit(rightY, rightX, 2)
        except TypeError:
            '''
            leftLineCoeficients = self.lastLeftLineCoeficients
            rightLineCoeficients = self.lastRightLineCoeficients
            '''
            leftLineCoeficients = np.array([0,0,0])
            rightLineCoeficients = np.array([0,0,0])
            
        return leftLineCoeficients, rightLineCoeficients
    
    def drawLinesAndLane(self, image, leftLineCoeficients, rightLineCoeficients, drawLines = True, drawLane = True):
        # Creates a copy of the input image where the lines will be drawn
        imageLines = image.copy()
        
        linesY = np.linspace(0, image.shape[0]-1, image.shape[0])
        

        leftLineX = leftLineCoeficients[0]*linesY**2 + leftLineCoeficients[1]*linesY + leftLineCoeficients[2]
        rightLineX = rightLineCoeficients[0]*linesY**2 + rightLineCoeficients[1]*linesY + rightLineCoeficients[2]
        
        '''
        self.lastLeftLineCoeficients = leftLineCoeficients
        self.lastRightLineCoeficients = rightLineCoeficients
        '''

        if drawLines == True:
            for i in range(0, image.shape[0] - 1):
                if ((leftLineX[i] > 0) & (leftLineX[i] < image.shape[1] - 1)): 
                    imageLines[np.int_(linesY[i]), np.int_(leftLineX[i]), 1:] = 255
                    imageLines[np.int_(linesY[i]), np.int_(leftLineX[i]) + 1, 1:] = 255
                    imageLines[np.int_(linesY[i]), np.int_(leftLineX[i]) - 1, 1:] = 255
                if ((rightLineX[i] > 0) & (rightLineX[i] < image.shape[1] - 1)): 
                    imageLines[np.int_(linesY[i]), np.int_(rightLineX[i]), 1:] = 255
                    imageLines[np.int_(linesY[i]), np.int_(rightLineX[i]) + 1, 1:] = 255
                    imageLines[np.int_(linesY[i]), np.int_(rightLineX[i]) - 1, 1:] = 255
        
        # If "drawLane" is true, the lane line gets drawn into the image. Feature which is useful when the image gets
        # unwarped and overlaped to the original undistorted image
        if drawLane == True:
            # Creates an empty image where the lane will be drawn
            imageLane = np.zeros_like(imageLines)
            # Generates a polygon indicating where the lane is located
            # leftLineBorder is an array of all the points (x,y) which belong to the left line organized upwards
            leftLineBorder = np.array([np.transpose(np.vstack([leftLineX, linesY]))])
            # rightLineBorder is an array of all the points (x,y) which belong to the right line, but this time organized downwards
            rightLineBorder = np.array([np.flipud(np.transpose(np.vstack([rightLineX, linesY])))])
            # lanePoints is an array of all the points which define the polygon drawing the lane
            lanePoints = np.hstack((leftLineBorder, rightLineBorder))
            
            # Draws the lane into the blank image created before
            cv2.fillPoly(imageLane, np.int_([lanePoints]), (0,255,0))
            # Adds the image to the one meant to be returned
            imageLaneAndLines = cv2.addWeighted(imageLines, 0.5, imageLane, 1, 0)
        else:
            imageLaneAndLines = imageLines
            
        return imageLaneAndLines
        
    
    def getMeterPolynoms(self, leftX, leftY, rightX, rightY, X_mPerPix = (3.7/700), Y_mPerPix = (30/720)):
        # Gets the polynomic coeficients of the lane lines in metric values
        try:
            leftLineCoeficientsMeters = np.polyfit(leftY*Y_mPerPix, leftX*X_mPerPix, 2)
            rightLineCoeficientsMeters = np.polyfit(rightY*Y_mPerPix, rightX*X_mPerPix, 2)
        except TypeError:
            '''
            leftLineCoeficientsMeters = self.lastLeftLineCoeficientsMeters
            rightLineCoeficientsMeters = self.lastRightLineCoeficientsMeters
            '''
            leftLineCoeficientsMeters = np.array([0,0,0])
            rightLineCoeficientsMeters = np.array([0,0,0])
        
        return leftLineCoeficientsMeters, rightLineCoeficientsMeters
    
    def calculateCurvatureMeters(self, leftLineCoeficientsMeters, rightLineCoeficientsMeters, Y_mPerPix = (30/720), Ymax = 719):
        # Gets the curvature of the left and right lines 
        leftLineRad = ((1+(2*leftLineCoeficientsMeters[0]*Ymax*Y_mPerPix + leftLineCoeficientsMeters[1])**2)**(3/2))/abs(2*leftLineCoeficientsMeters[0])
        rightLineRad = ((1+(2*rightLineCoeficientsMeters[0]*Ymax*Y_mPerPix + rightLineCoeficientsMeters[1])**2)**(3/2))/abs(2*rightLineCoeficientsMeters[0])

        return leftLineRad, rightLineRad
    
    def calculateVehiclePos(self, leftLineCoeficientsMeters, rightLineCoeficientsMeters, X_mPerPix = (3.7/700), Y_mPerPix = (30/720), Xmax = 1279, Ymax = 719):
        # Gets the position of the car, relative to the center of the image. Positive values indicate that the car is leaned to the right, 
        # Negative values indicate that the car is leaned to the left
        Xcenter = (Xmax/2)*X_mPerPix
        XleftLine = leftLineCoeficientsMeters[0]*(Ymax*Y_mPerPix)**2 + leftLineCoeficientsMeters[1]*(Ymax*Y_mPerPix) + leftLineCoeficientsMeters[2]
        XrightLine = rightLineCoeficientsMeters[0]*(Ymax*Y_mPerPix)**2 + rightLineCoeficientsMeters[1]*(Ymax*Y_mPerPix) + rightLineCoeficientsMeters[2]
        carWidth = XrightLine - XleftLine
        XcarCenter = XleftLine + carWidth/2
        Xcar = XcarCenter - Xcenter
        
        return Xcar
        
    def addDataToOriginal(self, originalImage, laneLinesImage, leftLineRad, rightLineRad, posCar):
        outputImage = cv2.addWeighted(originalImage, 1, laneLinesImage, 0.4, 0)
        stringLeftLineRad = "Left line radius: " + str(round(leftLineRad,2))
        stringRightLineRad = "Right line radius: " + str(round(rightLineRad,2))
        if posCar > 0:
            stringPosCar = "The car is " + str(abs(round(posCar,2))) + " m to the right"
        elif posCar < 0:
            stringPosCar = "The car is " + str(abs(round(posCar,2))) + " m to the left"
        elif posCar == 0:
            stringPosCar = "The car is at the center"
        
        # Prints the variables to the image in white text
        cv2.putText(outputImage, stringLeftLineRad, (20, 40), cv2.FONT_HERSHEY_DUPLEX, 1.5, (255,255,255))
        cv2.putText(outputImage, stringRightLineRad, (20, 100), cv2.FONT_HERSHEY_DUPLEX, 1.5, (255,255,255))
        cv2.putText(outputImage, stringPosCar, (20, 160), cv2.FONT_HERSHEY_DUPLEX, 1.5, (255,255,255))
        
        return outputImage
    

