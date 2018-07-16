## Writeup

**Advanced Lane Finding Project**

The steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients and sobel to create a thresholded binary image.
* Apply a perspective transform to rectify binary image.
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[distortion_image]: ./output_images/distortion_image.jpg "distortion_image"
[calibrate_image]: ./output_images/calibrate_image.jpg "calibrate_image"
[color_image]: ./output_images/color_image.jpg "color_image"
[binary_image]: ./output_images/binary_image.jpg "binary_image"
[unwarped_image]: ./output_images/binary_image.jpg "unwarped_image"
[warped_image]: ./output_images/warped_image.jpg "warped_image"
[curve_image]: ./output_images/curve_image.jpg "curve_image"
[add_weighted_image]: ./output_images/basic1.jpg "add_weighted_image"


## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/wangzhe0912/self_driver_project_2/blob/master/writeup.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is contained in the second code cell of the IPython notebook.

The IPython notebook can be found in './project/P2.ipynb'.

Beside the IPython notebook, I have written an python file: "./project/camera.py".

In this file, I have written a class Camera.

This class include 2 core method: load_picures and cal_undistort.

load_picures is used to load picture and compute the camera matrix and distortion coefficients.

After load picture, We can use cal_undistort to calibrate the distortion image.


### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

To demonstrate this step, I will describe how I apply the distortion correction to one of the test images like this one. This image is distortion image.

![alt text][distortion_image]

After calibrate the distortion image, the ouput image like this one.

![alt text][calibrate_image]

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

I used a combination of color and sobel thresholds to generate a binary image (thresholding steps at lines # through # in `./project/thresholding_filter.py`).  Here's an example.  
This is the origin image.

![alt text][color_image]

After thresholds filters, it generates a binary image:

![alt text][binary_image]

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform in the file `perspective_transform`.
This file includes three core function:

1. get_transform_m(src, dst)
2. perspective_transform(img, m)
3. reverse_picture(perspective_filter_img, left_fit, right_fit, Minv)

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

![alt text][unwarped_image]

After transforme, I get the output like this:

![alt text][warped_image]

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

Then I did some other stuff and fit my lane lines with a 2nd order polynomial kinda like this:
```python
# polyfit land line for left and right separately.
left_fit = np.polyfit(lefty, leftx, 2)
right_fit = np.polyfit(righty, rightx, 2)
```

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

I did this in my code in `fitting_curve.py`.
The core code as follows:

```python
# transform from real world and pixel in image
ym_per_pix = 30 / 720  # meters per pixel in y dimension
xm_per_pix = 3.7 / 700  # meters per pixel in x dimension
left_fit_cr = np.polyfit(lefty * ym_per_pix, leftx * xm_per_pix, 2)
# Calculate the new radii of curvature
y_eval = np.max(ploty)
left_curverad = ((1 + (2 * left_fit_cr[0] * y_eval * ym_per_pix + left_fit_cr[1]) ** 2) ** 1.5) / np.absolute(2 * left_fit_cr[0])

car_center = (left_fit[0] * 720 ** 2 + left_fit[1] * 720 + left_fit[2] +
              right_fit[0] * 720 ** 2 + right_fit[1] * 720 + right_fit[2]) / 2
image_center = binary_warped.shape[1] / 2
bias_meter = (car_center - image_center) * xm_per_pix
```

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step in lines in my code in `./project/pipeline.py` in the line 34 - 40.  Here is an example of my result on a test image:

![alt text][add_weighted_image]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

The pipeline for video is same as single image.
But there is a point need to pay attention: The first step compute the camera calibration matrix and distortion coefficients given a set of chessboard images just need to calculate for beginning. And when we process images, we can just use the same calibration matrix and distortion coefficients.

Here's a [link to my video result](./output_videos/project_video_output.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

In this project, I have these problems as following:

1. I don't know how to utilize the relation between continuous frame. Because of the function of clip1.fl_image(process_image) just receive an function as input, and this function just can receive current image as input, so I don't know how to use continuous frame information to current iframe image.
2. In challenge_video.mp4 and harder_challenge_video.mp4 file, I found the thresholds filters is too diffcult to get the precise land line. So I don't know how to improve the performance in challenge_video.mp4 and harder_challenge_video.mp4.


Now, about this project, I think I can do these to make it more robust:

1. Use the continuous frame information to improve the performance.
2. In fitting curve action, Instead of find the argmax in left part as leftx_base and argmax in right part as rightx_base, I think maybe we can consider the interval between leftx_base and rightx_base. Generally, the interval between leftx_base and rightx_base should be fixed. So we can find the max point in picture and calculated another base by fixed interval.
