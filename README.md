##Advanced Lane Finding Project

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

[image1]: ./output_images/undistort_output.png "Undistorted"
[image2]: ./output_images/road_undist.png "Road Transformed"
[image3]: ./output_images/binary_combo_example.png "Binary Example"
[image4]: ./output_images/warped_straight_lines.png "Warp Example"
[image6]: ./output_images/lane_and_poly_on_image.png "Output"
[video1]: ./output_images/project_video_with_lines.mp4 "Video"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points
###Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Advanced-Lane-Lines/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!
###Camera Calibration

####1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is contained in the `calibrate_camera.py` file.

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection. See lines 12 to 17.

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients (lines 23-38) using the `cv2.calibrateCamera()` function (line 47).  I applied this distortion correction to the test image using the `cv2.undistort()` function (line 50) and obtained this result:

![alt text][image1]

###Pipeline (single images)

####1. Provide an example of a distortion-corrected image.
To demonstrate this step, I will describe how I apply the distortion correction to one of the test images like this one:
![alt text][image2]
####2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.
I used a combination of color (S and V channels) and gradient (x and y) thresholds to generate a combined binary image (see the function called `pipeline()` in `process_image.py` at line #21).  Here's an example of my output for this step. I used this output to fine tune the threshold parameters to capture lines on the more difficult images by imagining the union of the two sets which are each an intersection of the gradients and colour channels respectively.

![alt text][image3]

####3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform includes a function called `warp_image()`, which appears at line #6 in the file `process_image.py`.  The `warp_image()` function takes as inputs an image (`img`). I chose, starting from the example in the `example_writeup.md` as recommended by one of the forum moderators and then fine tuning it until I was satisfied I had a good warped image. The hardcoded values are:

```
source = np.float32([[592, 451],
                     [230, 701],
                     [1078, 701],
                     [689, 451]])
dest = np.float32([[320, 0],
                   [320, 720],
                   [960, 720],
                   [960, 0]])

```

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

![alt text][image4]

####4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

To identify lane pixels I used convolutions to identify the peaks in the image. The fucntion `find_window_centroids` in `process_image.py` is called in `find_lanes.py` on line 43. The resulting points are stored in `window_centroids`. The function `find_window_centroids` takes the following parameters:
```
# window settings
window_width = 50
window_height = 90 # Break image into 9 vertical layers since image height is 720
margin = 40 # How much to slide left and right for searching

```

The values of these parameters were determined by starting with those provided in the lesson 'Sliding Window Search' and tuning them through experimentation on all test images.

Then in lines 45-52 in `find_lanes.py` I fit a polynomial to the discovered points. Lines 65-67 generate the points for plotting. Lines 69-114 do the following:
* Compare the curvature of the current fit to the previous and, if it's off by more than an amount `tolerance`, consider that this is not an accurate line. Do this for both the left and right lines.
* Lines 80 and 81 keep a fixed number of previous fitted x points for the left lane (we do this again for the right)
* Line 83 calculates the average of the last `average window` points and stores them for comparison later in `left_lane.bestx` (which is an object of the Class `Line` found in `process_image.py`). We do the same for the right line.
* In lines 101-102 we assign the newly calculated rolling averaged fitted points to the lines for this iteration.
* If the lane lines - left or right - are considered to be outside tolerances:
** Do not add these points to the rolling average
** Assign the previous rolling average to the fitted line

You'll see the outcome in a later figure in 6. below.

####5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

The radius of curvature is calculated in lines 54-63 in `find_lines.py`. Centre of the lane and the car's offset from centre are calculated in 128-137.

####6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step in lines 116-149 in `find_lanes.py`.  Here is an example of my result on a test image:

![alt text][image6]

---

###Pipeline (video)

####1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result][video1]

---

###Discussion

####1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

The main problem I face is the line following the average curve instead of the actual. This results in the furthest end of the curve crossing over into the other lane. What I would still need to work on to improve this:
* Instead of comparing the curve of the line to the previous curve and using that to determine whether the line is a good fit or not I could use the discovered points and compare those to previous.
* Discard found points that are outliers and instead only use those within a certain tolerance of the previous lines

