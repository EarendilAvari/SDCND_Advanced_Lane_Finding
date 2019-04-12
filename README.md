# Self driving car nanodegree
## Project 2: Advanced lane line finding

### Description

This is the second project of the Udacity's nanodegree in autonomous vehicles. In this project advanced concepts of computer vision are used in order to detect lines on the street through a camera and extract some information. For that the library OpenCV is used, together with other Python standard libraries.

For the project a pipeline of functions was developed to be applied first in images and after that in videos. 

The computer vision concepts used on this project are:

- Camera calibration
- Perspective transformation
- Image gradients
- Color spaces HSL and HSV

### Development process on images

#### Camera calibration
The first step in order to extract reliable data from images is to undistort the images taken from it. In order to do that the matrix of the camera and the distortion coeficients are needed. Both of them can be obtained doing a camera calibration process.

In order to calibrate the camera, 20 images of a chess board with different levels and types of distortion where used, as well with different view points.

For that, I created the function "Calibrate" within a class "Camera" (located in the file "Camera.py"). This function takes as input the path where the images are found, together with the number of columns and rows of chessboard corners. 

The function finds these chessboard points in every calibration image using the OpenCV function "findChessboardCorners" and relates its position on the image with points of the "real world" with format (x,y,z), located on a plane with coordinate z = 0. For example, a chessboard point found in the position (230, 480) of an image would be mapped to the point (2,3,0), what means that it corresponds to the second point of the third row of chessboard points on an image.

 Then, it uses the OpenCV function "calibrateCamera" with the location of the points in the image and in the chessboard points matrix and returns the camera matrix and the distortion coeficients.
 
Having the camera matrix and the distortion coeficients, any image can be undistorted using the function "UndistortImage" of the class "Camera". This function uses the OpenCV function "undistort" and returns an undistorted image.

One of the calibration images before and after the calibration can be seen here:

![ Image1](/mnt/data/Udacity/Self_driving_car_engineer/Project2/My_project/ImgsReport/01_CalibImgAfterUndistort.png  "Camera calibration")

By displaying both images (undistorted and distorted) it can be seen how the undistorting processing worked on the camera. In the case of this camera, the original images are not that distorted, only in the edges a convex distortion can be seen, which is corrected on the undistorted image.

#### Undistort street images
The first step of the pipeline is to undistort the image of the street using the function "UndistortImage". 

![ Image2](/mnt/data/Udacity/Self_driving_car_engineer/Project2/My_project/ImgsReport/02_straight_lines1_beforeAndAfterCalib.png  "Undistorted street image")

By plotting this test image it can be seen that a little bit of the original image is missing on the undistorted image. Also some objects are bigger, but the difference is actually minimal.

#### Binary image for line detection

##### Gradient with grayscale image

This is the trickiest part of this project because it involves testing a lot in order to get a binary image where the lane lines can be seen and the rest is ideally filtered.

First I tried getting the gradient of an image. The gradient corresponds to the change rate of a multidimensional function and in the case of images can be defined as:

$$\triangledown f = \begin{pmatrix}
g _{x}  \\
g _{y}
\end{pmatrix} = \begin{pmatrix}
\frac{\delta f}{\delta x}  \\
\frac{\delta f}{\delta y}
\end{pmatrix}$$

Where:

$ \frac{\delta f}{\delta x} $ : Derivative with respect to x (gradient in the x direction)
$ \frac{\delta f}{\delta y} $ : Derivative with respect to y (gradient in the y direction)

The gradient needs to be applied separately to different channels of an image. Since we are looking for the edges of the lines without thinking about their color, the most straightforward way to get what we want is applying the gradient on a grayscale version of the original image.

On the pixels where the edges of the lane lines are located the gradient in the x direction is high because they represent a big change in the x direction. Since the lines are diagonally displayed on an image of a camera located at the front of a car, the gradient in the y direction will be also high on the edge of the lines.

This gives the idea that getting the magnitude of the gradient, where both directions are considered can be also usefull. The magnitude of the gradient is defined by:

$$ | \triangledown f | = \sqrt{ g _{x} ^{2} + g _{y} ^{2} }$$

The gradient in the direction X or Y of an image can be calculated by applying to it the Sobel operator. As ilustration, the Sobel operator in direction X with kernel size 3 looks like this:

$$ S _{x} = \begin{pmatrix}
-1  &  0  & +1 \\
-2 & 0 & +2 \\
-1 & 0 & +1
\end{pmatrix}$$

In order to apply the Sobel operator, the convolution of it with the image is calculated:

$$ G _{x} = S _{x} * Image $$

This operation can be done with the OpenCV function "Sobel".

A binary image can be created by selecting the pixels where the gradient is within a threshold range. In the next image, this is applied on the image "straight_lines1.jpg" after it being undistorted using the function "Sobel".

![ Image3](/mnt/data/Udacity/Self_driving_car_engineer/Project2/My_project/ImgsReport/04_BinImgsGrayGradient.png "Undistorted street image")

By showing all three images can be seen that the lines are better shown on the binary image created by taking the gradient in the X direction. The difference is not that big though. But the gradient in the X direction also sucesses better on filtering the other elements on the image.

Another approach is to create a binary image by calculating the gradient's direction (not to be confused with the gradient on the X direction or in the Y direction). This can be calculated as following:

$$ \Theta = \tan ^{-1} \Big ( \frac {g _{y}}{g _{x}} \Big ) $$

![ Image4](/mnt/data/Udacity/Self_driving_car_engineer/Project2/My_project/ImgsReport/05_BinImgsGrayGradientDir.png "Undistorted street image")


The binary image created by calculating the direction of the gradient shows a lot of noise, but with very good defined lines. This can be useful to mask another binary image, but it will not work as standalone binary image.

##### Gradient with HSL color space

As said before, the gradient of an image needs to be calculated on a channel of it. It is known that a lane line is normally very bright, so, what if we could use that property to get better binary images?

This can be done by exploring the HSL and HSV color spaces.

![ Image5](/mnt/data/Udacity/Self_driving_car_engineer/Project2/My_project/ImgsReport/24_ColorSpaces.png "Color spaces")

The HSL and HSV color spaces are other forms to represent an image as digital data different from the traditional RGB format. For this application the space HSL is more interesting, since we can make use of the component L which indicates the "Lightness" of a pixel. Let's see how the HSL components of our test image "straight_lines1.jpg" look like:

![ Image6](/mnt/data/Udacity/Self_driving_car_engineer/Project2/My_project/ImgsReport/08_imgStraightLinesHSLGray.png "Color spaces")

Note: here are the components shown as grayscale.

It can be seen that the component L is basically the same than a grayscale image, while the component S is like a "darkened" version of a grayscale image, where some details are deleted. The component H is more intriguing since it represents the core color of a pixel.

The next figure shows binary images of "straight_lines1.jpg" in the direction X, Y and its magnitude using the different channels of the HSL color space.

![ Image6](/mnt/data/Udacity/Self_driving_car_engineer/Project2/My_project/ImgsReport/06_BinImgsHSLGradient.png "Binary images with HSL")

By showing all the images received can be seen that the channel H is not usefull to draw the lines into the binary images, In it a lot of scenery is detected but the lines not. The channel S does a very good job by detecting the lines, but they are missing sometimes. There is not a big difference between taking the gradient in direction x, y or the magnitude.

On the other hand, the channel L does even a better job than the channel S by detecting the lines, but it also detects some scenery. It must be remembered that all that scenery will not be considered when the line pixels will be detected and calculated, so it could be safely said that the L channel is here the best option to detect the lines. 

Only to try, lets see what we get with the same settings but another theshold range

![ Image7](/mnt/data/Udacity/Self_driving_car_engineer/Project2/My_project/ImgsReport/07_BinImgsHSLGradientOtherThreshold.png "Binary images with HSL 2")


By increasing the threshold range, it gets more evident that the Channel L, with the gradient in the X direction is the best approach to detect the lines. 
They are drawn very clear and with not that many other irrelevant elements. The binary images created with the channel S are still missing some parts of the 
line.

This can be explained by thinking how the colorspace HSL works. The H component (Hue) corresponds to the value of the base color (red, green, blue, cyan, magenta and yellow), the S component (Saturation) corresponds to the strength of the shown color, and the L (Lightness) corresponds to how close the color is to white. Since the lane lines are normally white or light yellow, they are more easy to detect using the L component because they are very close to white. But what if the line is darker yellow? In that case the channel L can fail to detect the lane lines, but in the channel S, they will still be strong enough.

Another good idea to create a binary image is to use the HSL components purely.

![ Image8](/mnt/data/Udacity/Self_driving_car_engineer/Project2/My_project/ImgsReport/09_imgStraightLinesHSLBin.png "Binary images with HSL 2")

By showing the binary images created thresholding the HSL components can be seen how good the lines are detected by the L and S components. The L component finds still more pieces of the line, but also another parts of the image, so it would not be smart to use it as it is without applying gradient. Specially when the light is very strong, the L component could be very high everywhere, disturbing the line measurements a lot. In the S component with the selected threshold almost only the lines can be seen, so it seems viable to use it to detect the lines.

I defined as standard method to get the binary image as a combination between the gradient in the L channel in the X direction and a thresholed version of the S channel

Let's see how it works on other images:

![ Image9](/mnt/data/Udacity/Self_driving_car_engineer/Project2/My_project/ImgsReport/10_AllTestImagesBin01.png "Binary images general")

![ Image10](/mnt/data/Udacity/Self_driving_car_engineer/Project2/My_project/ImgsReport/11_AllTestImagesBin02.png "Binary images general")

By applying both methods on all images, can be seen that the combination of both is a very good approach to get the lane lines. What does not get detected by the gradient in direction X of the L component, gets detected with the S component and viceversa. This method will be used to get the lane line points and after that the equations corresponding to the lane lines.

A little optimization was here done with the parameters, the threshold range for the gradient in X was changed from (30, 150) to (35. 175) and the threshold range for the S component was changed from (180, 255) to (180, 250) in order to prevent very strong shadows to appear in the binary image, like in "test5.jpg"

To get binary images based on the gradient of an image I created the function "GradientCalc" in the class "BinaryImg" and to get binary images from pure HSL components I created the function "HSLBinary" in the class "BinaryImg". This can be found on the file "BinaryImg.py"

#### Perspective transformation

The next step in the lane line detection and measurement is to get a "bird view" image of the street. For this I programmed the function "WarpPolygonToSquare" in the class "Camera". This function moves the point of view of the camera in order to get an image where the lane lines can be seen parallel. This function uses the OpenCV functions "getPerspectiveTransform" and "warpPerspective". For the transformation, it uses a polygon with 4 vertices in the origin image and a square in the destiny image and it receives the Y position of the top and bottom points of the polygon and the X position of the four vertices.

By transforming "straight_lines1.jpg" to bird view using arbitrarly selected polygon vertices we get this:

![ Image11](/mnt/data/Udacity/Self_driving_car_engineer/Project2/My_project/ImgsReport/12_StraightLines1_transformed.png "Warped 1")

As it can be seen in the images, the perspective transform is working well getting an image in "bird view" perspective. The problem is that the lines are not parallel as they should be. That is because the vertices of the selected polygon were selected very roughly. That the lines are parallel in the bird view image is very important in order to get good calculations after that. So it is needed to get the vertices of that polygon in a precise way.

Here is where the work in the first project can be very useful to calculate those vertices. So we import from the first project functions "region_of_interest", "hough_vertices" and "hough_lines", now located in the file HoughLines.py and we update the "hough_lines" function to return the points at the beginning and at the end of the line. More information about this function on: https://github.com/EarendilAvari/SDCND_Finding_lane_lines

The values of the polygon calculated using this method are ytop = 460, ybottom = 720, xbottomleft = 207, xtopleft = 508, xbottomright = 1103, xtopright = 696. By warping the image again using this value we get this:

![ Image12](/mnt/data/Udacity/Self_driving_car_engineer/Project2/My_project/ImgsReport/13_StraightLines1_transformed_corr.png "Warped 2")

By using this method to find the vertices of the conversion polygon the lane lines in the converted image are completely paralel. What means that the conversion is now accurate and appropiate to be used. The best thing is, that this process is not needed in the pipeline, these vertices can be used for any image to convert it into bird view image, so they can be used as parameters within the pipeline.

![ Image13](/mnt/data/Udacity/Self_driving_car_engineer/Project2/My_project/ImgsReport/14_TestImagesWarped.png "Warped 3")

By showing the three images and their warped version it can be seen how well this polygon does performing the transformation. In all the three
images the lines stay parallel, which means that the conversion is valid and can be used for further analysis.



