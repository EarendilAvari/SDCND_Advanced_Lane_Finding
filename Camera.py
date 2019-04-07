#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr  7 00:33:36 2019

@author: earendilavari
"""

import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

class Camera:
    # Class with functions to calibrate the camera, undistort images, do perspective transform        
    def Calibrate(self, images, nx, ny):
        # Arrays to store object points and image points from all the images
        CalibObjPoints = [] # 3D points in real world image
        CalibImgPoints = [] # 2D points in image plane
        
        # Prepare object points, like (0,0,0), (1,0,0), (2,0,0) ... (7,5,0)
        CalibObjCoordinates = np.zeros((ny*nx, 3), np.float32) 
        CalibObjCoordinates[:,:2] = np.mgrid[0:nx,0:ny].T.reshape(-1,2) #x, y coordinates
        
        for imgName in images:
            # Read each image
            img = mpimg.imread(imgName)
            
            # Convert image to grayscale:
            imgGray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            
            # Find the chessboard corners:
            cornersFound, corners = cv2.findChessboardCorners(imgGray, (nx, ny), None)
            
            if cornersFound == True:
                CalibImgPoints.append(corners)
                CalibObjPoints.append(CalibObjCoordinates)
                
        # Now the OpenCV function calibrateCamera is used to get the camera matrix and the 
        # distortion coeficients (k1, k2, p1, p2, etc.)
        
        calibSucess, cameraMatrix, distCoeffs, rvecs, tvecs = cv2.calibrateCamera(CalibObjPoints, CalibImgPoints,
                                                                                  img.shape[1::-1], None, None)
        
        return cameraMatrix, distCoeffs
    
    def UndistortImage(self, image, cameraMatrix, distCoeffs):
        undistortedImg = cv2.undistort(image, cameraMatrix, distCoeffs, None, cameraMatrix)
        return undistortedImg
    
    def UndistortAndSaveImageList(self, images, cameraMatrix, distCoeffs, savePath):
        for i in range(0, len(images) - 1):
            # Read each image
            img = mpimg.imread(images[i])
            undistortedImg = cv2.undistort(img, cameraMatrix, distCoeffs, None, cameraMatrix)
            undistortedImgFile = savePath % (i+1)
            cv2.imwrite(undistortedImgFile, undistortedImg)
            
            


        
        
    