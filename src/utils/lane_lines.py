import numpy as np

from src.utils.params import *
from src.utils.env import settings
from src.models.line import Line


def find_lane_lines(img_binary):
    if settings.left_glines.detected & settings.right_glines.detected:
        search_around_poly(img_binary)
    else:
        print('Using sliding windows')
        find_lane_lines_sliding_windows(img_binary)


# To make the algorithm faster we only going to use this function the first time  or when the lane line lost
def find_lane_lines_sliding_windows(img_binary):
    img_binary = np.copy(img_binary)
    # Take a histogram of the bottom half of the image
    histogram = np.sum(img_binary[img_binary.shape[0] // 2:, :], axis=0)

    # Find the peak of the left and right halves of the histogram these will be the starting point
    midpoint = np.int(histogram.shape[0] // 2)
    left_x_base = np.argmax(histogram[:midpoint])
    right_x_base = np.argmax(histogram[midpoint:]) + midpoint

    # Set height of windows - based on nwindows above and image shape
    window_height = np.int(img_binary.shape[0] // SW_NWINDOWS)

    # Identify the x and y positions of all nonzero(activated) pixels in the image
    nonzero = img_binary.nonzero()
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

    # process the inds
    process_inds(img_binary.shape, nonzero_x, nonzero_y, left_lane_inds, right_lane_inds)


def search_around_poly(img_binary):
    img_binary = np.copy(img_binary)
    # Grab activated pixels
    nonzero = img_binary.nonzero()
    nonzero_y = np.array(nonzero[0])
    nonzero_x = np.array(nonzero[1])

    sa_margin = int(SW_MARGIN // 2)

    # Set the area of search based on activated x-values
    # within the +/- margin of our polynomial function
    left_lane_inds = find_pixels_around(nonzero_x, nonzero_y, settings.left_glines.current_fit, sa_margin)
    right_lane_inds = find_pixels_around(nonzero_x, nonzero_y, settings.right_glines.current_fit, sa_margin)

    # Process the inds
    process_inds(img_binary.shape, nonzero_x, nonzero_y, left_lane_inds, right_lane_inds)


def process_inds(image_shape, nonzero_x, nonzero_y, left_lane_inds, right_lane_inds):
    # Extract left and right line pixel positions
    left_x = nonzero_x[left_lane_inds]
    left_y = nonzero_y[left_lane_inds]
    right_x = nonzero_x[right_lane_inds]
    right_y = nonzero_y[right_lane_inds]

    # Calculate the polynomial for both lines
    left_fit = np.polyfit(left_y, left_x, 2)
    right_fit = np.polyfit(right_y, right_x, 2)

    # Generate x and y values for plotting
    ploty = np.linspace(0, image_shape[0] - 1, image_shape[0])
    left_fit_x = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
    right_fit_x = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]

    # Calculate the pos
    y_max = np.argmax(ploty)
    img_xmid = image_shape[1] / 2
    left_pos = (img_xmid - left_fit_x[y_max]) * XM_PER_PIX
    right_pos = (right_fit_x[y_max] - img_xmid) * XM_PER_PIX

    # Create the result left lane line
    settings.left_glines.detected = True
    settings.left_glines.append_fit(left_fit)
    settings.left_glines.append_x_fitted(left_fit_x)
    settings.left_glines.append_pos(left_pos)
    settings.left_glines.ally = ploty
    settings.left_glines.calculate_curvature()

    # Create the result right lane line
    settings.right_glines.detected = True
    settings.right_glines.append_fit(right_fit)
    settings.right_glines.append_x_fitted(right_fit_x)
    settings.right_glines.append_pos(right_pos)
    settings.right_glines.ally = ploty
    settings.right_glines.calculate_curvature()

    settings.offset = calculate_distance_from_center(
        img_xmid, settings.left_glines.best_x[y_max], settings.right_glines.best_x[y_max]
    )

    sanit_check(settings.left_glines, settings.right_glines)


def sanit_check(left_lines: Line, right_lines: Line):
    left_lines.detected = left_lines.check_sanity_radius() & left_lines.check_sanity_pos()
    right_lines.detected = right_lines.check_sanity_radius() & right_lines.check_sanity_pos()


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


def find_pixels_around(nonzero_x, nonzero_y, current_fit, margin):
    return ((nonzero_x > validate_pixels_nonzero(nonzero_y, current_fit, -margin)) &
            (nonzero_x < validate_pixels_nonzero(nonzero_y, current_fit, margin)))


def validate_pixels_nonzero(nonzero, current_fit, margin):
    return current_fit[0] * (nonzero ** 2) + current_fit[1] * nonzero + current_fit[2] + margin


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
