from src.utils.cam_cal import *
from src.utils.image_utils import *
from src.utils.lane_lines import *


def process_image(p_image):
    """
    Pipeline with the complete image process
    The goals / steps of this project are the following:
    * Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
    * Apply a distortion correction to raw images.
    * Use color transforms, gradients, etc., to create a threshold binary image.
    * Apply a perspective transform to rectify binary image ("birds-eye view").
    * Detect lane pixels and fit to find the lane boundary.
    * Determine the curvature of the lane and vehicle position with respect to center.
    * Warp the detected lane boundaries back onto the original image.
    * Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.
    """
    # 1. Un-distort the image to get better measure
    un_dist = undistort_image(p_image, settings.mtx, settings.dist)
    # 2. Apply the advanced transform to get the final image
    img_adv = advanced_transform(
        un_dist, kernel_size=KERNEL_SIZE, s_thresh=S_CHANNEL_THRESH, sx_thresh=SOBEL_X_THRESH
    )
    # 3. Get the bird eye view and the invert transform matrix
    binary_warped, minv = bird_eye_transform(img_adv, settings.aoi_src, settings.bird_dst)
    # 4. Find the lane lines in the bird eye view, determine the curvature of the lane and vehicle position
    find_lane_lines(binary_warped)
    # 5. Draw the lines in the un-distorted image
    result = draw_final_lines(
        binary_warped, minv, un_dist, settings.left_glines, settings.right_glines, settings.offset
    )
    return result
