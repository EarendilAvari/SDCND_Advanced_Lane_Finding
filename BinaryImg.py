#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr  7 12:42:31 2019

@author: earendilavari
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

class BinaryImg:
    def GradientCalc(self, startImage, imgType = 'grayscale', imgChannel = 'l', calcType = 'dirX', kernelSize = 3, thresh = (0, 255)):
        # Based on the parameter imgType, it is decided if the image is transformed to grayscale, to HSL or to HSV
        if imgType == 'grayscale':
            procImage = cv2.cvtColor(startImage, cv2.COLOR_RGB2GRAY)
            # In the case of grayscale, it only has one channel, so it only can be worked on it
            procChannel = procImage
        elif imgType == 'HSL':
            procImage = cv2.cvtColor(startImage, cv2.COLOR_RGB2HLS)
            if imgChannel == 'h':
                procChannel = procImage[:,:,0]
            elif imgChannel == 'l':
                procChannel = procImage[:,:,1]
            elif imgChannel == 's':
                procChannel = procImage[:,:,2]
            else:
                print("Channel of image invalid. Expected 'h', 's' or 'l'")
                return startImage
        elif imgType == 'HSV':
            procImage = cv2.cvtColor(startImage, cv2.COLOR_RGB2HSV)
            if imgChannel == 'h':
                procChannel = procImage[:,:,0]
            elif imgChannel == 's':
                procChannel = procImage[:,:,1]
            elif imgChannel == 'v':
                procChannel = procImage[:,:,2]
            else:
                print("Channel of image invalid. Expected 'h', 's' or 'v'")
                return startImage
        else:
            print("Type of image value invalid. Expected 'grayscale', 'HSL' or 'HSV")
            return startImage
        
        # Based on the parameter calcType, it decides to calculate the gradient on the X axis, on the Y axis,
        # its magnitude or its direction
        if calcType == 'dirX':
            gradientImage = cv2.Sobel(procChannel, cv2.CV_64F, 1, 0, ksize = kernelSize)
            absGradientImg = abs(gradientImage)
            imgGradientScaled = np.uint8(255*absGradientImg/np.max(absGradientImg))
        elif calcType == 'dirY':
            gradientImage = cv2.Sobel(procChannel, cv2.CV_64F, 0, 1, ksize = kernelSize)
            absGradientImg = abs(gradientImage)
            imgGradientScaled = np.uint8(255*absGradientImg/np.max(absGradientImg))
        elif calcType == 'magnitude':
            gradientImageX = cv2.Sobel(procChannel, cv2.CV_64F, 1, 0, ksize = kernelSize)
            gradientImageY = cv2.Sobel(procChannel, cv2.CV_64F, 0, 1, ksize = kernelSize)
            magGradientImg = np.sqrt(np.power(gradientImageX, 2) + np.power(gradientImageY, 2))
            imgGradientScaled = np.uint8(255*magGradientImg/np.max(magGradientImg))
        elif calcType == 'direction':
            gradientImageX = cv2.Sobel(procChannel, cv2.CV_64F, 1, 0, ksize = kernelSize)
            gradientImageY = cv2.Sobel(procChannel, cv2.CV_64F, 0, 1, ksize = kernelSize)
            absGradientImgX = abs(gradientImageX)
            absGradientImgY = abs(gradientImageY)
            gradientDir = np.arctan2(absGradientImgY, absGradientImgX)
            imgGradientScaled = gradientDir
            
        else:
            print("Calculation type invalid. Expected 'dirX', 'dirY', 'magnitude' or 'direction'")
            return startImage
        
        gradientImgThresholded = np.zeros_like(imgGradientScaled)
        gradientImgThresholded[(imgGradientScaled > thresh[0]) & (imgGradientScaled < thresh[1])] = 1
        
        return gradientImgThresholded
    

        


        
        
            