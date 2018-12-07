import cv2
import matplotlib.pyplot as plt
import numpy as np

from src.utils.params import *
from src.models.line import Line


# To make the algorithm faster we only going to use this function the first time  or when the lane line lost
def find_lane_lines_sliding_windows(img_binary_bird_eye):
    # Take a histogram of the bottom half of the image
    histogram = np.sum(img_binary_bird_eye[img_binary_bird_eye.shape[0] // 2:, :], axis=0)

    # Create an output image to draw on and visualize the result
    out_img = np.dstack((img_binary_bird_eye, img_binary_bird_eye, img_binary_bird_eye)) * 255

    # Find the peak of the left and right halves of the histogram these will be the starting point
    midpoint = np.int(histogram.shape[0] // 2)
    left_x_base = np.argmax(histogram[:midpoint])
    right_x_base = np.argmax(histogram[midpoint:]) + midpoint

    # Set height of windows - based on nwindows above and image shape
    window_height = np.int(img_binary_bird_eye.shape[0] // SW_NWINDOWS)

    # Identify the x and y positions of all nonzero(activated) pixels in the image
    nonzero = img_binary_bird_eye.nonzero()
    nonzero_y = np.array(nonzero[0])
    nonzero_x = np.array(nonzero[1])

    # Current positions to be updated later for each window in nwindows
    left_x_current = left_x_base
    right_x_current = right_x_base

    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []

    # iterate over the nwindows
    for window in range(SW_NWINDOWS):
        """ window dot
        b-----c
        |     |
        a-----d
        """
        lwa = (int(left_x_current - (SW_MARGIN / 2)), int((SW_NWINDOWS - window) * window_height))
        lwb = (lwa[0], lwa[1] - window_height)
        lwc = (lwb[0] + SW_MARGIN, lwb[1])
        lwd = (lwc[0], lwa[1])

        rwa = (int(right_x_current - (SW_MARGIN / 2)), int((SW_NWINDOWS - window) * window_height))
        rwb = (rwa[0], rwa[1] - window_height)
        rwc = (rwb[0] + SW_MARGIN, rwb[1])
        rwd = (rwc[0], rwc[1])

        # Draw the window on the visualization image
        cv2.rectangle(out_img, lwa, lwc, (0, 255, 0), 2)
        cv2.rectangle(out_img, rwa, rwc, (0, 255, 0), 2)

        # Identify the nonzero pixels inside the the window
        inds_left = ((nonzero_y >= lwb[1]) & (nonzero_y < lwa[1]) &
                     (nonzero_x >= lwb[0]) & (nonzero_x < lwd[0])).nonzero()[0]
        inds_right = ((nonzero_y >= rwb[1]) & (nonzero_y < rwa[1]) &
                      (nonzero_x >= rwb[0]) & (nonzero_x < rwd[0])).nonzero()[0]

        # Append indices to the lists
        left_lane_inds.append(inds_left)
        right_lane_inds.append(inds_right)

        # If the number of pixels you found in Step before are greater than the parameter min_pix
        # re-center our window based on the mean position of these pixels.
        if len(inds_left) > SW_MIN_PIX:
            left_x_current = np.int(np.mean(nonzero_x[inds_left]))
        if len(inds_right) > SW_MIN_PIX:
            right_x_current = np.int(np.mean(nonzero_x[inds_right]))

    # Concatenate the arrays of indices (previously was a list of lists of pixels)
    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)

    # Extract left and right line pixel positions
    left_x = nonzero_x[left_lane_inds]
    left_y = nonzero_y[left_lane_inds]
    right_x = nonzero_x[right_lane_inds]
    right_y = nonzero_y[right_lane_inds]

    # Calculate the polynomial for both lines
    left_fit = np.polyfit(left_y, left_x, 2)
    right_fit = np.polyfit(right_y, right_x, 2)

    # Generate x and y values for plotting
    ploty = np.linspace(0, img_binary_bird_eye.shape[0] - 1, img_binary_bird_eye.shape[0])
    left_fit_x = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
    right_fit_x = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]

    # Calculate the polynomial in real meters
    left_fit_cr = np.polyfit(ploty * YM_PER_PIX, left_fit_x * XM_PER_PIX, 2)
    right_fit_cr = np.polyfit(ploty * YM_PER_PIX, right_fit_x * XM_PER_PIX, 2)

    # Calculate the offset from the center
    y_max = np.argmax(ploty)
    offset = calculate_distance_from_center(midpoint, left_fit_x[y_max], right_fit_x[y_max])
    # print(offset)
    
    # Calculate the radius of curvature of the line
    left_curved, right_curved = measure_curvature_real(y_max, left_fit_cr, right_fit_cr)
    # print(left_curved, 'm', right_curved, 'm')
    
    # Create the result left lane line
    left_lane = Line()
    left_lane.detected = True
    left_lane.current_fit = left_fit
    left_lane.radius_of_curvature = left_curved
    left_lane.allx = left_fit_x
    left_lane.ally = ploty

    # Create the result right lane line    
    right_lane = Line()
    right_lane.detected = True
    right_lane.current_fit = right_fit
    right_lane.radius_of_curvature = right_curved
    right_lane.allx = right_fit_x
    right_lane.ally = ploty

    # out_img[left_y, left_x] = [255, 0, 0]
    # out_img[right_y, right_x] = [0, 0, 255]

    # plt.imshow(out_img)
    # plt.plot(left_fit_x, ploty, color='yellow')
    # plt.plot(right_fit_x, ploty, color='yellow')
    # plt.xlim(0, 1280)
    # plt.ylim(720, 0)
    # plt.show()

    return left_lane, right_lane, offset


# def find_pixels_around():
#     return ((nonzero_x > (left_fit[0] * (nonzero_y ** 2) + left_fit[1] * nonzero_y +
#                    left_fit[2] - SW_MARGIN)) & (nonzero_x < (left_fit[0] * (nonzero_y ** 2) +
#                                                              left_fit[1] * nonzero_y + left_fit[
#                                                                  2] + SW_MARGIN)))


# Fit a polynomial to all the relevant pixels you've found in your frame
def fit_poly(img_shape, left_x, left_y, right_x, right_y):
    # Fit a second order polynomial to each with np.polyfit()
    left_fit = np.polyfit(left_y, left_x, 2)
    right_fit = np.polyfit(right_y, right_x, 2)

    # Generate x and y values for plotting
    ploty = np.linspace(0, img_shape[0] - 1, img_shape[0])

    # Calc both polynomials using ploty, left_fit and right_fit
    left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
    right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]

    return left_fitx, right_fitx, ploty


def search_around_poly(img_binary, left_fit, right_fit):
    # Grab activated pixels
    nonzero = img_binary.nonzero()
    nonzero_y = np.array(nonzero[0])
    nonzero_x = np.array(nonzero[1])

    sa_margin = int(SW_MARGIN // 2)

    # Set the area of search based on activated x-values
    # within the +/- margin of our polynomial function ###
    # Hint: consider the window areas for the similarly named variables
    # in the previous quiz, but change the windows to our new search area
    left_lane_inds = ((nonzero_x > (left_fit[0] * (nonzero_y ** 2) + left_fit[1] * nonzero_y +
                                    left_fit[2] - sa_margin)) & (nonzero_x < (left_fit[0] * (nonzero_y ** 2) +
                                                                              left_fit[1] * nonzero_y + left_fit[
                                                                                  2] + sa_margin)))
    right_lane_inds = ((nonzero_x > (right_fit[0] * (nonzero_y ** 2) + right_fit[1] * nonzero_y +
                                     right_fit[2] - sa_margin)) & (nonzero_x < (right_fit[0] * (nonzero_y ** 2) +
                                                                                right_fit[1] * nonzero_y + right_fit[
                                                                                    2] + sa_margin)))

    # Again, extract left and right line pixel positions
    left_x = nonzero_x[left_lane_inds]
    left_y = nonzero_y[left_lane_inds]
    right_x = nonzero_x[right_lane_inds]
    right_y = nonzero_y[right_lane_inds]

    # Fit new polynomials
    left_fit_x, right_fit_x, ploty = fit_poly(img_binary.shape, left_x, left_y, right_x, right_y)

    # Visualization
    # Create an image to draw on and an image to show the selection window
    out_img = np.dstack((img_binary, img_binary, img_binary)) * 255
    window_img = np.zeros_like(out_img)
    # Color in left and right line pixels
    out_img[nonzero_y[left_lane_inds], nonzero_x[left_lane_inds]] = [255, 0, 0]
    out_img[nonzero_y[right_lane_inds], nonzero_x[right_lane_inds]] = [0, 0, 255]

    # Generate a polygon to illustrate the search window area
    # And recast the x and y points into usable format for cv2.fillPoly()
    left_line_window1 = np.array([np.transpose(np.vstack([left_fit_x - sa_margin, ploty]))])
    left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_fit_x + sa_margin,
                                                                    ploty])))])
    left_line_pts = np.hstack((left_line_window1, left_line_window2))
    right_line_window1 = np.array([np.transpose(np.vstack([right_fit_x - sa_margin, ploty]))])
    right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fit_x + sa_margin,
                                                                     ploty])))])
    right_line_pts = np.hstack((right_line_window1, right_line_window2))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(window_img, np.int_([left_line_pts]), (0, 255, 0))
    cv2.fillPoly(window_img, np.int_([right_line_pts]), (0, 255, 0))
    result = cv2.addWeighted(out_img, 1, window_img, 0.3, 0)

    # Plot the polynomial lines onto the image
    plt.plot(left_fit_x, ploty, color='yellow')
    plt.plot(right_fit_x, ploty, color='yellow')
    plt.imshow(result)
    plt.show()

    return result


def measure_curvature_real(y_eval, left_fit_cr, right_fit_cr):
    """
    Calculates the curvature of polynomial functions in meters.
    :param y_eval:
    :param left_fit_cr:
    :param right_fit_cr:
    :return:
    """
    # Calculation of R_curve (radius of curvature)
    left_curved = ((1 + (2 * left_fit_cr[0] * y_eval * YM_PER_PIX + left_fit_cr[1]) **
                    2) ** 1.5) / np.absolute(2 * left_fit_cr[0])
    right_curved = ((1 + (2 * right_fit_cr[0] * y_eval * YM_PER_PIX + right_fit_cr[1]) **
                     2) ** 1.5) / np.absolute(2 * right_fit_cr[0])

    return left_curved, right_curved


def calculate_distance_from_center(img_xmid, bx_left, bx_right):
    lane_center = bx_right - bx_left / 2
    offset = (lane_center - img_xmid) * XM_PER_PIX
    return offset
