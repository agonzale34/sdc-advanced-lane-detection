import numpy as np
import cv2

from src.models.line import Line


def grayscale(image):
    # create a copy to work locally
    img = np.copy(image)
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)


# Function that thresholds the S-channel of HLS
def hls_select(image, thresh=(0, 255)):
    # create a copy to work locally
    img = np.copy(image)
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    s_channel = hls[:, :, 2]
    binary_output = np.zeros_like(s_channel)
    binary_output[(s_channel > thresh[0]) & (s_channel <= thresh[1])] = 1
    return binary_output


# Function that applies Sobel x or y,
def abs_sobel_thresh(image, orient='x', sobel_kernel=3, thresh=(0, 255)):
    gray = grayscale(image)
    # Apply x or y gradient with the OpenCV Sobel() function
    # and take the absolute value
    abs_sobel = 0
    if orient == 'x':
        abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel))
    if orient == 'y':
        abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel))
    # Rescale back to 8 bit integer
    scaled_sobel = np.uint8(255 * abs_sobel / np.max(abs_sobel))
    # Create a copy and apply the threshold
    binary_output = np.zeros_like(scaled_sobel)
    # Here I'm using inclusive (>=, <=) thresholds, but exclusive is ok too
    binary_output[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 1
    # Return the result
    return binary_output


# Function that combine sobel and channel s transforms
def advanced_transform(img, kernel_size=3, s_thresh=(170, 255), sx_thresh=(20, 100)):
    s_binary = hls_select(img, thresh=s_thresh)
    sxbinary = abs_sobel_thresh(img, orient='x', sobel_kernel=kernel_size, thresh=sx_thresh)
    # Combine the two binary thresholds
    combined_binary = np.zeros_like(sxbinary)
    combined_binary[(s_binary == 1) | (sxbinary == 1)] = 1
    return combined_binary


# Function that pply a perspective transform to rectify binary image ("birds-eye view")
def bird_eye_transform(image, src, dst):
    img_size = (image.shape[1], image.shape[0])
    m = cv2.getPerspectiveTransform(src, dst)
    minv = cv2.getPerspectiveTransform(dst, src)
    warped = cv2.warpPerspective(image, m, img_size, flags=cv2.INTER_LINEAR)
    return warped, minv


def draw_final_lines(img_warped, p_matrix, img_un_dist, left_lane: Line, right_lane: Line, offset):
    # Create an image to draw the lines on
    warp_zero = np.zeros_like(img_warped).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_lane.best_x, left_lane.ally]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_lane.best_x, right_lane.ally])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))

    # Warp the blank back to original image space using inverse perspective matrix (p_matrix)
    newwarp = cv2.warpPerspective(color_warp, p_matrix, (img_un_dist.shape[1], img_un_dist.shape[0]))

    # curvature
    radius = int(left_lane.radius_of_curvature)  # + right_lane.radius_of_curvature) // 2
    radius_tx = 'Radius of Curvature = ' + str(radius) + '(m)'
    offset_tx = 'Vehicle is ' + str(round(offset, 2)) + 'm left of center'

    # add text
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img_un_dist, radius_tx, (50, 50), font, 1, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(img_un_dist, offset_tx, (50, 100), font, 1, (255, 255, 255), 2, cv2.LINE_AA)

    # Combine the result with the original image
    return cv2.addWeighted(img_un_dist, 1, newwarp, 0.3, 0)
