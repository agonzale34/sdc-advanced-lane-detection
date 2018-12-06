from src.utils.cam_cal import *
from src.utils.plt_utils import *
from src.utils.image_utils import *
from src.utils.params import *
from src.models.settings import *
import matplotlib.image as mpimg
import numpy as np
import cv2
import matplotlib.pyplot as plt

# testing the pipeline
cam_cal_path = '../resources/camera_cal/'
# 1. calibrate the camera
# mtx, dist = calibrate_camera(9, 6, cam_cal_path, 'cali*.jpg')
# Read in the saved mtx and dist
pickle = pickle.load(open(cam_cal_path + CC_FILE, 'rb'))

settings = Settings()

settings.mtx = pickle[CC_MTX]
settings.dist = pickle[CC_DIST]
settings.aoi_xmid = 0.511
settings.aoi_ymid = 0.625
settings.aoi_upsz = 0.08
settings.aoi_upds = 15
settings.aoi_basesz = 0.390

image = mpimg.imread('../resources/test_images/test2.jpg')
# image = mpimg.imread('../resources/test_images/straight_lines2.jpg')
settings.find_aoi_src_dst(image)

un_dist = undistort_image(image, settings.mtx, settings.dist)

# Apply the advanced transform to get the final image
img_adv = advanced_transform(
    un_dist, kernel_size=KERNEL_SIZE, s_thresh=S_CHANNEL_THRESH, sx_thresh=SOBEL_X_THRESH
)

# Get the bird eye view
binary_warped = bird_eye_transform(img_adv, settings.aoi_src, settings.bird_dst)

# Assuming you have created a warped binary image called "binary_warped"
# Take a histogram of the bottom half of the image
histogram = np.sum(binary_warped[binary_warped.shape[0] // 2:, :], axis=0)
# Create an output image to draw on and visualize the result
out_img = np.dstack((binary_warped, binary_warped, binary_warped)) * 255
# Find the peak of the left and right halves of the histogram
# These will be the starting point for the left and right lines
midpoint = np.int(histogram.shape[0] // 2)
leftx_base = np.argmax(histogram[:midpoint])
rightx_base = np.argmax(histogram[midpoint:]) + midpoint
# HYPERPARAMETERS
# Choose the number of sliding windows
nwindows = 9
# Set the width of the windows +/- margin
margin = 200
# Set minimum number of pixels found to recenter window
minpix = 50

# Set height of windows - based on nwindows above and image shape
window_height = np.int(binary_warped.shape[0] // nwindows)
# Identify the x and y positions of all nonzero (i.e. activated) pixels in the image
nonzero = binary_warped.nonzero()
nonzeroy = np.array(nonzero[0])
nonzerox = np.array(nonzero[1])
# Current positions to be updated later for each window in nwindows
leftx_current = leftx_base
rightx_current = rightx_base

# Create empty lists to receive left and right lane pixel indices
left_lane_inds = []
right_lane_inds = []

# iterate over the nwindows
# Step through the windows one by one
for window in range(nwindows):
    """ window dot
    b-----c
    |     |
    a-----d
    """
    lwa = (int(leftx_current - (margin / 2)), int((nwindows - window) * window_height))
    lwb = (lwa[0], lwa[1] - window_height)
    lwc = (lwb[0] + margin, lwb[1])
    lwd = (lwc[0], lwa[1])

    rwa = (int(rightx_current - (margin / 2)), int((nwindows - window) * window_height))
    rwb = (rwa[0], rwa[1] - window_height)
    rwc = (rwb[0] + margin, rwb[1])
    rwd = (rwc[0], rwc[1])

    # Draw the window on the visualization image
    cv2.rectangle(out_img, lwa, lwc, (0, 255, 0), 2)
    cv2.rectangle(out_img, rwa, rwc, (0, 255, 0), 2)

    # Identify the nonzero pixels inside the the window
    ins_left = ((nonzeroy >= lwb[1]) & (nonzeroy < lwa[1]) & (nonzerox >= lwb[0]) & (nonzerox < lwd[0])).nonzero()[0]
    ins_right = ((nonzeroy >= rwb[1]) & (nonzeroy < rwa[1]) & (nonzerox >= rwb[0]) & (nonzerox < rwd[0])).nonzero()[0]

    # Append indices to the lists
    left_lane_inds.append(ins_left)
    right_lane_inds.append(ins_right)

    # If the number of pixels you found in Step before are greater than your hyperparameter minpix
    # re-center our window based on the mean position of these pixels.
    if len(ins_left) > minpix:
        leftx_current = np.int(np.mean(nonzerox[ins_left]))
    if len(ins_right) > minpix:
        rightx_current = np.int(np.mean(nonzerox[ins_right]))

# Concatenate the arrays of indices (previously was a list of lists of pixels)
left_lane_inds = np.concatenate(left_lane_inds)
right_lane_inds = np.concatenate(right_lane_inds)

# Extract left and right line pixel positions
leftx = nonzerox[left_lane_inds]
lefty = nonzeroy[left_lane_inds]
rightx = nonzerox[right_lane_inds]
righty = nonzeroy[right_lane_inds]
# Assuming we have `left_fit` and `right_fit` from `np.polyfit` before
left_fit = np.polyfit(lefty, leftx, 2)
right_fit = np.polyfit(righty, rightx, 2)
# Generate x and y values for plotting
ploty = np.linspace(0, binary_warped.shape[0] - 1, binary_warped.shape[0])
left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]

out_img[lefty, leftx] = [255, 0, 0]
out_img[righty, rightx] = [0, 0, 255]

plt.imshow(out_img)
plt.plot(left_fitx, ploty, color='yellow')
plt.plot(right_fitx, ploty, color='yellow')
plt.xlim(0, 1280)
plt.ylim(720, 0)
plt.show()
