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

[image1]: ./output_images/undistorted_calibration2.jpg "Undistorted calibration matrix"
[image2]: ./output_images/undistorted_test1.jpg "Road Transformed"
[image3]: ./output_images/bintest3_2.jpg "Binary Example"
[image4]: ./output_images/binary_wraped.jpg "Warp Example"
[image5]: ./output_images/test2.jpg "Fit Visual"
[image6]: ./output_images/example_output.jpg "Output"
[image7]: ./output_images/undistorted-warped.png "Undistorted wraped"
[image8]: ./output_images/original_undistorted.jpg "Original undistorted"
[video1]: ./processed_project_video.mp4 "Video project"
[video2]: ./challenge_video_processed.mp4 "Challenge video"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is contained in lines 9 through 48 of  the file called `Utility.py`. 

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result: 

![alt text][image1]
The final calibration matrices are saved in the pickle file 'camera_cal/wide_dist_pickle.p' 

### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

Using the camera calibration and distortion coefficients in 'camera_cal/wide_dist_pickle.p', through the function loadCoefficients(), located in the file called `Utility.py` I undistort the input image calling the function undistortImage(image, mtx, dist) in lines 57 through 59. I apply the distortion correction to one of the test images like this one:
![alt text][image2]

#### 2. Thresholded binary image

I used a combination of color and gradient thresholds to generate a binary image thresholding steps at lines 6 through 48 in `Thresholding.py`.  Here's an example of my output for this step.  

![alt text][image3]

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform includes a function called `warper()`, which appears in lines 3 through 12 in the file `imageWarper.py` .  The `warper()` function takes as inputs an image (`img`), as well as source (`src`) and destination (`dst`) points.  I chose the hardcode the source and destination points in the following manner:

```python
	# Define 4 source points
	self.src = np.float32([[250, self.imgSize[0]-25], [575, 460], 
				  [700, 460], [1150, self.imgSize[0]-25]])
	# Define 4 destination points
	self.dst = np.float32([[320, self.imgSize[0]-25], [320, 0], 
				  [960, 0], [960, self.imgSize[0]-25]])
```

This resulted in the following source and destination points:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 250, 695      | 320, 695      | 
| 575, 460      | 320, 0        |
| 700, 460      | 960, 0        |
| 1150,695      | 960, 695      |

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

![alt text][image7]


#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

Given the warped binary image from the previous step, I now fit a 2nd order polynomial to both left and right lane lines. In particular, I perform the following:

* Calculate a histogram of the bottom half of the image
* Partition the image into 9 horizontal slices
* Starting from the bottom slice, enclose a 200 pixel wide window around the left peak and right peak of the histogram (split the histogram in half vertically)
* Go up the horizontal window slices to find pixels that are likely to be part of the left and right lanes, recentering the sliding windows opportunistically
* Given 2 groups of pixels (left and right lane line candidate pixels), fit a 2nd order polynomial to each group, which represents the estimated left and right lane lines

I define class LaneLines which contalin left, right lanes in file 'Line.py'

![alt text][image5]

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

Given the polynomial fit for the left and right lane lines, I calculated the radius of curvature for each line according to formulas presented [here](http://www.intmath.com/applications-differentiation/8-radius-curvature.php). I also converted the distance units from pixels to meters, assuming 30 meters per 720 pixels in the vertical direction, and 3.7 meters per 700 pixels in the horizontal direction.
I implemented this step in lines 208 through 244 in my code in `Line.py` in the function `drawLane()`.

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step in lines 246 through 266 in my code in `Line.py` in the function `drawLane()`.  Here is an example of my result on a test image:

![alt text][image6]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [video1](./processed_project_video.mp4)
Here's a [video2] (./challenge_video_processed.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?
This is an simple version of computer vision based lane finding. There are multiple scenarios where this lane finder would fail. For example, the Udacity harder challenge video includes roads with different perspective which could be slam our perspective transform (see 'harder_challenge_video_processed.mp4'). Also, it is possible that other vehicles in front would trick the lane finder into thinking it was part of the lane. More work can be done to make the lane detector more robust, e.g. [deep-learning-based semantic segmentation](https://arxiv.org/pdf/1605.06211.pdf) to find pixels that are likely to be lane markers (then performing polyfit on only those pixels).
