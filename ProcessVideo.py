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
        # Deque of the last 100 polynomic coeficients 
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
            if len(self.listLastCoefs) > 10:
                self.listLastCoefs.popleft()
    def determineBestCoefs(self):
        ponderation = np.linspace(1,len(self.listLastCoefs),len(self.listLastCoefs))
        sumBestCoefs = np.array([0,0,0], dtype='float')
        sumPonderation = 0
        for i in range(0, len(self.listLastCoefs) - 1):
            sumBestCoefs += np.multiply(ponderation[i],self.listLastCoefs[i])
            sumPonderation += ponderation[i]
        self.bestCoefs = sumBestCoefs/sumPonderation

leftLine = Line()
rightLine = Line()


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

    return imageOutput, leftCoef, rightCoef, radLeft, radRight, posCar

def getNewLaneLines(image, camMatrix, distCoeff, warpParameters, coefsLastLeftLines, coefsLastRightLines, gradXLThresh = (35, 180), SThresh = (180, 250)):
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
    LeftX, LeftY, RightX, RightY, pixelsBinWarpedImage = linesProc.findNewLaneLinesPixels(binWarpedImage, coefsLastLeftLines, coefsLastRightLines,
                                                                                         paintLinePixels = True)
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

    return imageOutput, leftCoef, rightCoef, radLeft, radRight, posCar

def getLaneLinesEnhanced(image, camMatrix, distCoeff, warpParameters, leftLineObj, rightLineObj,
                         gradXLThresh = (35, 180), LThresh = (0, 255), SThresh = (180, 250)):
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
    # Get the lane lines pixels from the warped binary image using the function "findFirstLaneLinesPixels" of the class "LinesProcessing". Paints the left
    # line pixels blue and the right line pixels red
    if ((len(leftLineObj.listLastCoefs) > 10) & (len(rightLineObj.listLastCoefs) > 10)): 
        coefsLastLeftLines = leftLineObj.bestCoefs
        coefsLastRightLines = rightLineObj.bestCoefs
        LeftX, LeftY, RightX, RightY, pixelsBinWarpedImage = linesProc.findNewLaneLinesPixels(binWarpedImage, coefsLastLeftLines, coefsLastRightLines,
                                                                                         paintLinePixels = True)
    else:
        LeftX, LeftY, RightX, RightY, pixelsBinWarpedImage = linesProc.findFirstLaneLinesPixels(binWarpedImage, showWindows = False, paintLinePixels = True)    
    
    # Get the coeficients of second grade polynoms which defines the lines with the function "getLines" of the class "LinesProcessing". If wanted draws
    # the lane lines and/or the lane into the image.
    leftCoef, rightCoef, laneBinWarpedImage = linesProc.getLines(pixelsBinWarpedImage,LeftX,LeftY,RightX,RightY, drawLines = True, drawLane = True)
    # Get the coeficients of second grade polynoms which defines the lines in real measurement units with the function "getMeterPolynoms" of the class
    # "LinesProcessing" to be used in order to calculate the radius of curvature of the lines and the position of the car relative to the center.
    
    leftCoefMetric, rightCoefMetric = linesProc.getMeterPolynoms(LeftX,LeftY,RightX,RightY)
    
    resetSearch = False
    
    if ((np.all(leftCoef == [0,0,0]) == True) & (np.all(leftCoefMetric == [0,0,0]) == True)):
        leftLineObj.linesNotFoundQt += 1
        if (leftLineObj.linesNotFoundQt > 5):
            leftLineObj.linesNotFoundQt = 0
            resetSearch = True
        else:
            leftCoef = leftLineObj.lastCoefs
            leftCoefMetric = leftLineObj.lastMetricCoefs
    
    if ((np.all(rightCoef == [0,0,0]) == True) & (np.all(rightCoefMetric == [0,0,0]) == True)):
        rightLineObj.linesNotFoundQt += 1
        if (rightLineObj.linesNotFoundQt > 5):
            rightLineObj.linesNotFoundQt = 0
            resetSearch = True
        else:
            rightCoef = rightLineObj.lastCoefs
            rightCoefMetric = rightLineObj.lastMetricCoefs
            
    # If resetSearch is true, search the pixels from the beginning again on the same frame:
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
    # the same parameters used to warp the image on the step 5.
    
    # Updates the coeficients inside of the object
    leftLineObj.currentCoefs = leftCoef
    rightLineObj.currentCoefs = rightCoef
    leftLineObj.currentMetricCoefs = leftCoefMetric
    rightLineObj.currentMetricCoefs = rightCoefMetric
    leftLineObj.radiusCurvature = radLeft
    rightLineObj.radiusCurvature = radRight    
    
    laneUnwarpedImage = cam.UnwarpSquareToPolygon(laneBinWarpedImage, warpParameters[0], warpParameters[1], warpParameters[2], 
                                                  warpParameters[3], warpParameters[4], warpParameters[5])
    # Overlap the unwarped image into the original undistorted image using the function "addDataToOriginal" of the class "LinesProcessing". It also
    # prints the radius of curvature of both lines and the position of the car into the output image.
    imageOutput = linesProc.addDataToOriginal(undistImg, laneUnwarpedImage, radLeft, radRight, posCar)

    return imageOutput

def ProcessFrame(image, matrix, dist, wParameters):
    if ((len(leftLine.listLastCoefs) > 5) & (len(rightLine.listLastCoefs) > 5)):
        outputFrame, leftCoefs, rightCoefs, leftRadius, rightRadius, positionCar = getNewLaneLines(image, matrix, dist, wParameters, leftLine.bestCoefs, rightLine.bestCoefs,
                                                                                                   gradXLThresh = (35, 180), SThresh = (180, 250))
    else:
        outputFrame, leftCoefs, rightCoefs, leftRadius, rightRadius, positionCar = getStartLaneLines(image, matrix, dist, wParameters, 
                                                                                                     gradXLThresh = (35, 180), SThresh = (180, 250))
    leftLine.currentCoefs = leftCoefs
    rightLine.currentCoefs = rightCoefs
    leftLine.radiusCurvature = leftRadius 
    rightLine.radiusCurvature = rightRadius
    
    leftLine.calcDifference()
    rightLine.calcDifference()
    leftLine.addNewCoeficients(leftLine.radiusCurvature)
    rightLine.addNewCoeficients(rightLine.radiusCurvature)
    
    leftLine.determineBestCoefs()
    rightLine.determineBestCoefs()
    
    coefsDiffLeft.append(leftLine.coefsDiff)
    coefsDiffRight.append(rightLine.coefsDiff)
    
    leftLine.lastCoefs = leftLine.currentCoefs
    rightLine.lastCoefs = leftLine.currentCoefs
        
    return outputFrame
        

def ProcessFrameEnhanced(image, matrix, dist, wParameters):
    
    outputFrame = getLaneLinesEnhanced(image, matrix, dist, wParameters, leftLine, rightLine, gradXLThresh = (15, 180), 
                                       LThresh = (185, 255), SThresh = (100, 250))
    
    leftLine.calcDifference()
    rightLine.calcDifference()
    leftLine.addNewCoeficients(leftLine.radiusCurvature)
    rightLine.addNewCoeficients(rightLine.radiusCurvature)
    
    leftLine.determineBestCoefs()
    rightLine.determineBestCoefs()
    
    coefsDiffLeft.append(leftLine.coefsDiff)
    coefsDiffRight.append(rightLine.coefsDiff)
    
    leftLine.lastCoefs = leftLine.currentCoefs
    rightLine.lastCoefs = leftLine.currentCoefs
        
    return outputFrame
        
        
    
wrapParameters = [460, 720, 207, 601, 1103, 673]
cameraMatrix = np.array([[  1.15777829e+03,   0.00000000e+00,   6.67113866e+02],
                          [  0.00000000e+00,   1.15282230e+03,   3.86124658e+02],
                          [  0.00000000e+00,   0.00000000e+00,   1.00000000e+00]])
    
distCoefs = np.array([[-0.24688572, -0.02372825, -0.00109832,  0.00035104, -0.00260431]])



from moviepy.editor import VideoFileClip

def ProcessImage(image):
    result = ProcessFrameEnhanced(image, cameraMatrix, distCoefs, wrapParameters)
    return result

# %%

clip1 = VideoFileClip("project_video.mp4")
# clip1 = VideoFileClip("test_videos/solidWhiteRight.mp4")
white_clip = clip1.fl_image(ProcessImage).subclip(0,5)#NOTE: this function expects color images!!
white_output = 'project_video_output_5sec.mp4'
white_clip.write_videofile(white_output, audio=False)

'''
By applying the pipeline on the normal video it can be seen that the pipeline works well on it. The lines are always detected in the correct place and the 
radius of curvature of both lines looks also well.
'''

# %%

clip1 = VideoFileClip("challenge_video.mp4")
# clip1 = VideoFileClip("test_videos/solidWhiteRight.mp4")
white_clip = clip1.fl_image(ProcessImage) #NOTE: this function expects color images!!
white_output = 'challenge_video_output.mp4'
white_clip.write_videofile(white_output, audio=False)
    

# %% Challenge 2 

clip3 = VideoFileClip("harder_challenge_video.mp4")
# clip1 = VideoFileClip("test_videos/solidWhiteRight.mp4")
white_clip = clip3.fl_image(ProcessImage) #NOTE: this function expects color images!!
white_output = 'harder_challenge_video_output.mp4'
white_clip.write_videofile(white_output, audio=False)
    
    
    