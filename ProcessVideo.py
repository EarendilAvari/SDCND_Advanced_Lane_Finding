#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  9 18:37:35 2019

@author: earendilavari
"""
import numpy as np
import cv2
from collections import deque

exec(open('Camera.py').read())
exec(open('BinaryImg.py').read())
exec(open('ProcessLines.py').read())

cam = Camera()
binImg = BinaryImg()
linesProc = LinesProcessing()

coefsDiffRight = []
coefsDiffLeft = []


# Define a class to receive the characteristics of each line detection
class Line():
    def __init__(self):
        # Deque of the last 20 polynomic coeficients 
        self.listLastCoefs = deque()
        #polynomial coefficients averaged over the last n iterations
        self.bestCoefs = np.array([0,0,0], dtype='float')  
        #polynomial coefficients for the most recent fit
        self.currentCoefs = np.array([0,0,0], dtype='float')
        #polynomial metric coefficients for the most recent fit
        self.currentMetricCoefs = np.array([0,0,0], dtype='float')
        #polynomial coeficients for the last fit
        self.lastCoefs = np.array([0,0,0], dtype='float')
        #polynomial coeficients for the last metric fit
        self.lastMetricCoefs = np.array([0,0,0], dtype='float')
        #radius of curvature of the line in some units
        self.radiusCurvature = 0 
        #distance in meters of vehicle center from the line
        self.posCar = 0 
        #difference in fit coefficients between last coeficients and best found coeficients
        self.coefsDiff = np.array([0,0,0], dtype='float') 
        #Quantity of frames where a line could not be fitted
        self.linesNotFoundQt = 0
        
    def calcDifference(self):
        self.coefsDiff = np.abs(self.currentCoefs - self.bestCoefs)
    def addNewCoeficients(self, radiusOtherLine, maxDifference = 500):
        # Adds the polynomic coeficients of the last iteration if the radius of
        # curvature of the lines is smaller than maxDifference
        if (
                (abs(self.radiusCurvature - radiusOtherLine) < maxDifference) & 
                ((len(self.listLastCoefs) < 10) | ((self.coefsDiff[0] < 1e-03) & (self.coefsDiff[1] < 1) & (self.coefsDiff[2] < 1e+03)))
            ):
            self.listLastCoefs.append(self.currentCoefs)
            if len(self.listLastCoefs) > 20:
                self.listLastCoefs.popleft()
    def determineBestCoefs(self):
        ponderation = np.linspace(1,len(self.listLastCoefs),len(self.listLastCoefs))
        sumBestCoefs = np.array([0,0,0], dtype='float')
        sumPonderation = 0
        for i in range(0, len(self.listLastCoefs) - 1):
            sumBestCoefs += np.multiply(ponderation[i],self.listLastCoefs[i])
            # sumBestCoefs += self.listLastCoefs[i]
            sumPonderation += ponderation[i]
        self.bestCoefs = sumBestCoefs/sumPonderation
        # self.bestCoefs = sumBestCoefs/len(self.listLastCoefs)

leftLine = Line()
rightLine = Line()

def getLaneLinesEnhanced(image, camMatrix, distCoeff, warpParameters, leftLineObj, rightLineObj,
                         gradXLThresh = (35, 180), LThresh = (0, 255), SThresh = (100, 250), frameNumber = 0):
    
    if not hasattr(getLaneLinesEnhanced, "badLinesCounter"):
        getLaneLinesEnhanced.badLinesCounter = 0
    
    # Undistort the image using the function "UndistortImage" of "Camera" and the camera matrix and distortion coeficients obtained with 
    # the camera calibration.
    undistImg = cam.UndistortImage(image, camMatrix, distCoeff)
    # Obtain a binary image of the undistorted image by calculating the gradient in direction X of its L color channel using the function 
    # "GradientCalc" of the class "BinaryImg".
    binImgGradX_Lch = binImg.GradientCalc(undistImg, imgType = 'HSL', imgChannel = 'l', calcType = 'dirX', kernelSize = 3, thresh = gradXLThresh)
    binImg_Lch = binImg.HSLBinary(undistImg, imgChannel = 'l', thresh = LThresh)
    binImgGradX_Lch_masked = cv2.bitwise_and(binImgGradX_Lch, binImg_Lch)
    
    # Obtain a binary image of the undistorted image by thresholding its S color channel using the function "HSLBinary" of the class "BinaryImg".
    binImg_Sch = binImg.HSLBinary(undistImg, imgChannel = 's', thresh = SThresh)
    
    
    # Combine both binary images with the function "CombineBinaries" of the class "BinaryImg"
    binImage = binImg.CombineBinaries(binImgGradX_Lch_masked, binImg_Sch)
    # Warp the combined binary image into a "bird view" image with the function "WarpPolygonToSquare" of the class "Camera" using the polygon parameters
    # calculated with the function "hough_lines" created for the first project.
    binWarpedImage = cam.WarpPolygonToSquare(binImage, warpParameters[0], warpParameters[1], warpParameters[2], 
                                             warpParameters[3], warpParameters[4], warpParameters[5])
    # Do the deques of "good" coefficients for both lines have more than 5 elements?
    if ((len(leftLineObj.listLastCoefs) > 5) & (len(rightLineObj.listLastCoefs) > 5)): 
        # Get pixels of right and left lane lines based on the weighted average of the last good coeficients using "findNewLaneLinesPixels"
        coefsLastLeftLines = leftLineObj.bestCoefs
        coefsLastRightLines = rightLineObj.bestCoefs
        LeftX, LeftY, RightX, RightY, pixelsBinWarpedImage = linesProc.findNewLaneLinesPixels(binWarpedImage, coefsLastLeftLines, coefsLastRightLines,
                                                                                              distCenter = 75, paintLinePixels = True)
        # DATALOG FOR DEBUGGING
        logString = 'Frame N°: ' + str(frameNumber) + ' findNewLaneLinesPixels used with last best coeficients\n'
        pipelineLogger.write(logString)
    else:
        # Get pixels of right and left lane lines from the beginning using "findFirstLaneLinesPixels"
        LeftX, LeftY, RightX, RightY, pixelsBinWarpedImage = linesProc.findFirstLaneLinesPixels(binWarpedImage, showWindows = False, paintLinePixels = True)   
        logString = 'Frame N°: ' + str(frameNumber) + ' findFirstLaneLinesPixels used\n'
        pipelineLogger.write(logString)
    
    # Get polynomial coefficients based on the lane lines pixels found using "getPolynoms"
    leftCoef, rightCoef = linesProc.getPolynoms(LeftX,LeftY,RightX,RightY)
    
    # Get the coeficients of second grade polynoms which defines the lines in real measurement units with the function "getMeterPolynoms" of the class
    # "LinesProcessing" to be used in order to calculate the radius of curvature of the lines and the position of the car relative to the center.
    leftCoefMetric, rightCoefMetric = linesProc.getMeterPolynoms(LeftX,LeftY,RightX,RightY)
    
    resetSearch = False
    
    # If the calculated polynomial coefficients of the right line are too different of the ones of the left line and if the coefficients of the 
    # last frame are not 0 (it is not the very first frame) and the search close to last best line was performed (findNewLaneLinesPixels) used    
    if (((abs(leftCoef[0] - rightCoef[0]) > 1e-03) | (abs(leftCoef[1] - rightCoef[1]) > 1)) & 
    (leftLineObj.lastCoefs[0] != 0) & (leftLineObj.lastCoefs[1] != 0) & (leftLineObj.lastCoefs[2] != 0) &
    (rightLineObj.lastCoefs[0] != 0) & (rightLineObj.lastCoefs[1] != 0) & (rightLineObj.lastCoefs[1] != 0) & 
    ((len(leftLineObj.listLastCoefs) > 5) & (len(rightLineObj.listLastCoefs) > 5))):
        getLaneLinesEnhanced.badLinesCounter += 1
        # If this did not happen in more than 30 frames consecutively
        if (getLaneLinesEnhanced.badLinesCounter <= 30):
            leftCoef = leftLineObj.lastCoefs.copy()
            leftCoefMetric = leftLineObj.lastMetricCoefs.copy()
            rightCoef = rightLineObj.lastCoefs.copy()
            rightCoefMetric = rightLineObj.lastMetricCoefs.copy()
            logString = 'Frame N°: ' + str(frameNumber) + ', Coeficients for both lines of last frame used because new lines are too different from each other\n'
            pipelineLogger.write(logString)
        else:
            getLaneLinesEnhanced.badLinesCounter = 0
            resetSearch = True
            logString = 'Frame N°: ' + str(frameNumber) + ', Search arround last lines restarted. In more than 30 consecutive frames (1 s) a new line would be too different of the last one\n'
            pipelineLogger.write(logString)
    else:
        getLaneLinesEnhanced.badLinesCounter = 0
    # The function "getPolynoms" or "getMeterPolynoms" could not find a polynom for the left line
    if ((leftCoef[0] == 0) | (leftCoef[1] == 0) | (leftCoefMetric[0] == 0) | (leftCoefMetric[1] == 0)):
        leftLineObj.linesNotFoundQt += 1
        if (leftLineObj.linesNotFoundQt > 5):
            leftLineObj.linesNotFoundQt = 0
            resetSearch = True
            logString = 'Frame N°: ' + str(frameNumber) + ', Search arround last lines restarted. In more than 5 consecutive frames a polynom for the left line could not be created\n'
            pipelineLogger.write(logString)
        else:
            leftCoef = leftLineObj.lastCoefs.copy()
            leftCoefMetric = leftLineObj.lastMetricCoefs.copy()
            logString = 'Frame N°: ' + str(frameNumber) + ', Coeficients for left line of last frame used because a new polynom could not be created\n'
            pipelineLogger.write(logString)
    else:
        leftLineObj.linesNotFoundQt = 0
    
    # # The function "getPolynoms" or "getMeterPolynoms" could not find a polynom for the right line
    if ((rightCoef[0] == 0) | (rightCoef[1] == 0) | (rightCoefMetric[0] == 0) | (rightCoefMetric[1] == 0)):
        rightLineObj.linesNotFoundQt += 1
        if (rightLineObj.linesNotFoundQt > 5):
            rightLineObj.linesNotFoundQt = 0
            resetSearch = True
            logString = 'Frame N°: ' + str(frameNumber) + ', Search arround last lines restarted. In more than 5 consecutive frames a polynom for the right line could not be created\n'
            pipelineLogger.write(logString)
        else:
            rightCoef = rightLineObj.lastCoefs.copy()
            rightCoefMetric = rightLineObj.lastMetricCoefs.copy()
            logString = 'Frame N°: ' + str(frameNumber) + ', Coeficients for right line of last frame used because a new polynom could not be created\n'
            pipelineLogger.write(logString)
    else:
        rightLineObj.linesNotFoundQt = 0
          
    laneBinWarpedImage = linesProc.drawLinesAndLane(pixelsBinWarpedImage, leftCoef, rightCoef, drawLines = False, drawLane = True)    
    
    # If resetSearch is true, Restart the search of lane lines pixels from the beginning using "findFirstLaneLinesPixels". 
    # Get polynomial coefficients and draw lines with "getLines". Get metric polynomial coefficients using "getMeterPolynoms". 
    # Delete all elements of the deques of "good" coefficients
    if resetSearch == True:
        LeftX, LeftY, RightX, RightY, pixelsBinWarpedImage = linesProc.findFirstLaneLinesPixels(binWarpedImage, showWindows = False, paintLinePixels = True)
        leftCoef, rightCoef, laneBinWarpedImage = linesProc.getLines(pixelsBinWarpedImage,LeftX,LeftY,RightX,RightY, drawLines = True, drawLane = True)
        leftCoefMetric, rightCoefMetric = linesProc.getMeterPolynoms(LeftX,LeftY,RightX,RightY)
        # Cleans last coefs to search from the beginning again on the next frames:
        leftLineObj.listLastCoefs.clear()
        rightLineObj.listLastCoefs.clear()
    
    
    # Calculate the radius of curvature of both lines using the function "calculateCurvatureMeters" of the class "LinesProcessing". 
    radLeft, radRight = linesProc.calculateCurvatureMeters(leftCoefMetric, rightCoefMetric)
    # Calculate the position of the car relative to the center of the image using the function "calculateVehiclePos" of the class "LinesProcessing".
    posCar = linesProc.calculateVehiclePos(leftCoefMetric, rightCoefMetric)
    # Unwarp the image with the lane lines and the line drawn on it with the function "UnwarpSquareToPolygon" of the class "Camera" using
    # the same parameters used to warp the image before.
    laneUnwarpedImage = cam.UnwarpSquareToPolygon(laneBinWarpedImage, warpParameters[0], warpParameters[1], warpParameters[2], 
                                                  warpParameters[3], warpParameters[4], warpParameters[5])
    # Overlap the unwarped image into the original undistorted image using the function "addDataToOriginal" of the class "LinesProcessing". It also
    # prints the radius of curvature of both lines and the position of the car into the output image.
    imageOutput = linesProc.addDataToOriginal(undistImg, laneUnwarpedImage, radLeft, radRight, posCar)
    
    # Save polynomial coefficients and radius of curvature inside objects of class "line"
    leftLineObj.currentCoefs = leftCoef.copy()
    rightLineObj.currentCoefs = rightCoef.copy()
    leftLineObj.currentMetricCoefs = leftCoefMetric.copy()
    rightLineObj.currentMetricCoefs = rightCoefMetric.copy()
    leftLineObj.radiusCurvature = radLeft
    rightLineObj.radiusCurvature = radRight    
    

    return imageOutput        

def ProcessFrameEnhanced(image, matrix, dist, wParameters):
    
    # ONLY FOR DEBUGGING
    if not hasattr(ProcessFrameEnhanced, "frameNumber"):
        ProcessFrameEnhanced.frameNumber = 0
    ProcessFrameEnhanced.frameNumber += 1
    strFrame = 'Frame: ' + str(ProcessFrameEnhanced.frameNumber)
    
    # The parameters for the normal video are gradXLThresh = (35, 180), LThresh = (0, 255) and SThresh = (180, 250)
    # The parameters for the challenge video are gradXLThresh = (10, 180), LThresh = (190, 255), SThresh = (90, 250)
    
    outputFrame = getLaneLinesEnhanced(image, matrix, dist, wParameters, leftLine, rightLine, gradXLThresh = (35, 180), 
                                       LThresh = (0, 255), SThresh = (180, 250), frameNumber = ProcessFrameEnhanced.frameNumber)
    
    cv2.putText(outputFrame, strFrame, (20, 680), cv2.FONT_HERSHEY_TRIPLEX, 1, (255,255,0))
    
    # Calculate difference between last polynomial coefficients and average of good ones.
    leftLine.calcDifference()
    rightLine.calcDifference()
    
    # Add new "good" coefficients to deques. If deque is longer than 20 elements, delete last element and reorganize
    leftLine.addNewCoeficients(leftLine.radiusCurvature)
    rightLine.addNewCoeficients(rightLine.radiusCurvature)
    
    # Calculate weighed average of last "best" coefficients for both lines.
    leftLine.determineBestCoefs()
    rightLine.determineBestCoefs()
    
    # Save  coefficients of last frame to be used for the next one if necessary
    leftLine.lastCoefs = leftLine.currentCoefs.copy()
    rightLine.lastCoefs = rightLine.currentCoefs.copy()
    leftLine.lastMetricCoefs = leftLine.currentMetricCoefs.copy()
    rightLine.lastMetricCoefs = rightLine.currentMetricCoefs.copy()
        
    return outputFrame
        
        
    
wrapParameters = [460, 720, 207, 580, 1103, 696]
cameraMatrix = np.array([[  1.15777829e+03,   0.00000000e+00,   6.67113866e+02],
                          [  0.00000000e+00,   1.15282230e+03,   3.86124658e+02],
                          [  0.00000000e+00,   0.00000000e+00,   1.00000000e+00]])
    
distCoefs = np.array([[-0.24688572, -0.02372825, -0.00109832,  0.00035104, -0.00260431]])



from moviepy.editor import VideoFileClip

def ProcessImage(image):
    result = ProcessFrameEnhanced(image, cameraMatrix, distCoefs, wrapParameters)
    return result

# %%
pipelineLogger = open("Pipeline_logger.txt", "a")
clip1 = VideoFileClip("project_video.mp4")
# clip1 = VideoFileClip("test_videos/solidWhiteRight.mp4")
white_clip = clip1.fl_image(ProcessImage)#NOTE: this function expects color images!!
white_output = 'project_video_output.mp4'
white_clip.write_videofile(white_output, audio=False)
pipelineLogger.close()

#By applying the pipeline on the normal video it can be seen that the pipeline works well on it. The lines are always detected in the correct place and the 
#radius of curvature of both lines looks also well.


# %%
pipelineLogger = open("Pipeline_logger.txt", "w")
clip1 = VideoFileClip("challenge_video.mp4")
# clip1 = VideoFileClip("test_videos/solidWhiteRight.mp4")
white_clip = clip1.fl_image(ProcessImage) #NOTE: this function expects color images!!
white_output = 'challenge_video_output.mp4'
white_clip.write_videofile(white_output, audio=False)
pipelineLogger.close()
    

# %% Challenge 2 

pipelineLogger = open("Pipeline_logger.txt", "a")
clip3 = VideoFileClip("harder_challenge_video.mp4")
# clip1 = VideoFileClip("test_videos/solidWhiteRight.mp4")
white_clip = clip3.fl_image(ProcessImage) #NOTE: this function expects color images!!
white_output = 'harder_challenge_video_output.mp4'
white_clip.write_videofile(white_output, audio=False)
pipelineLogger.close()

    
    