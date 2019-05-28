## Writeup

---

**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./markup_images/checkboard_image.png "Undistorted"
[image2]: ./markup_images/sample_image.png "Road Transformed"
[image3]: ./markup_images/binary_straight.png "Binary Example"
[image4]: ./markup_images/birdview_straight.png "Warp Example"
[image5]: ./markup_images/curvature_eq.gif "Curvature Equation"
[image6]: ./markup_images/display_image.png "Output"
[video1]: ./output_images/project_video.mp4 "Video"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Advanced-Lane-Lines/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

The README is [here](./README.md). The writeUp is as follows.

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The camera calibration code is described in Section 2 of "advanced_lane_finding.ipynb", which describes the algorithm code snipet. 
And the code used in the pipeline is described at Camera.calibrate and Camera.undistort in "advanced_lane.py".

First, I calculate the camera intrinsic matrix K and distortion coefficients with the cv2.calibrateCamera function.
The inputs of the function are the checkerboard object points (objpoints) and the distorted grid points (imgpoints).

Using the calculated K, D matrix, I generate a checkerboard image with cv2.undistort. 
The undistort image of checker board image(calibration1.png) are as follows:

![alt text][image1]

### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

To demonstrate this step, I will describe how I apply the distortion correction to one of the test images(straight_lines1.jpg, test6.png) like this one:

![alt text][image2]

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

I describe a method of color transforms and gradient to generate a binarized image.
The binarization code is described in Section 3 of "advanced_lane_finding.ipynb". 
The code in the pipeline is described in "advanced_lane.py" in Camera.binalize.

The function converts the input RGB image to the HLS image, and extracts pixels with the higher value of S channel(saturation).
However, since the convert with S channel cannot detect faraway lanes where colors are averaged, I also use a sobel filter for the x axis.

The following shows the example of binary image.

![alt text][image3]

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code of the perspective transform is described in Section 4 of "advanced_lane_finding.ipynb". 
And the code used in the pipeline is described at Camera.perspective function in "advanced_lane.py".

An Affine transformation that specifies four corners is used in the perspective transform. 
The source coordinates(src_corners) were calculated manually based on the left and right 
lines of the "straight_line1.png" image. 
The destination coordinates(dst_corners) were projected onto the image with the same input image size,
and the x position of the left and right lines are placed at a distance separated by a margin(side_margin)
away from the image edges. The definitions of src_corners and dst_corners is as follow:

```python
src_corners = np.float32([
              [710, 467],
              [1108,719],
              [207, 719],
              [570, 467]])
dst_corners = np.float32([
              [img_w-side_margin, 0],
              [img_w-side_margin, img_h-1],
              [side_margin, img_h-1],
              [side_margin, 0]])
```

The following shows the calculated coordinates.

| Source        | Destination    | 
|:-------------:|:--------------:| 
| 710, 467      | 250, 0         | 
| 1108,719      | 250, 720       |
| 207, 719      | 1030, 720      |
| 570, 467      | 1030, 0        |

As shown in the following figure, I execute perspective transformation with the margins of 250 pixels.
The left and right lines are roughly parallel, so the perspective transformation works correctly.

![alt text][image4]

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

The lane detection code is described in section 5 of "advanced_lane_finding.ipynb".
And the code used in the pipeline is described at Line.detect function in "advanced_lane.py".

First, I create a histogram in which the frequency is a number of white pixels and the class is a column, 
and  divide into two at the image center.
And the frequency peak is calculated as the base position of the left and right lines.
I search the histogram window along the line from the bottom base position to the top of the image, 
and extract only the pixels of the lines.

Second, I execute the regression analysis of the quadratic function using the pixels of this line.
If the base positions of left and right lines are separated from the threshold(width_threshold) 
and the curvature is not separated from the threshold(radius_threshold),
the pixels of the lines are extracted using the lane detected in the previous frame.
If the above conditions are not met, the histogram approach is used.


#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

The code of curvature calculation is described in Section 6 of "advanced_lane_finding.ipynb". 
And the code used in the pipeline is described in Line.calc_curvature of "advanced_lane.py". 
The curvature of the left and right lanes is calculated as the following equation.

![alt text][image5]
  
#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

The code of the road area visualization is described in section 7 of "advanced_lane_finding.ipynb". 
And the code used in the pipeline is described at lane_boundaries_image function in "advanced_lane.py". 
The generated image is shown below.

![alt text][image6]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result][video1]

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

- problem1: the curvature is not stable because the curvature is calculated using single frame
    * average quadratic functions in multiple frames
- problem2: line extraction fails for large curves because the cutting position of the road is fixed.
    * extract the pixels of the lines using segmentation results.