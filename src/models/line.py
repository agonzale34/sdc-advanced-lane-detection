import numpy as np

from src.utils.params import *


# Define a class to receive the characteristics of each line detection
class Line:

    def __init__(self):
        # was the line detected in the last iteration?
        self.detected = False
        # x values of the last n fits of the line
        self.recent_x_fitted = []
        # average x values of the fitted line over the last n iterations
        self.best_x = None
        # polynomial coefficients averaged over the last n iterations
        self.recent_fit = []
        self.best_fit = None
        # polynomial coefficients for the most recent fit
        self.current_fit = [np.array([False])]
        # radius of curvature of the line in some units
        self.radius_of_curvature = None
        self.recent_radius = []
        self.best_radius = 0
        # distance in meters of vehicle center from the line
        self.line_base_pos = None
        self.recent_pos = []
        self.best_pos = 0
        # difference in fit coefficients between last and new fits
        self.diffs = np.array([0, 0, 0], dtype='float')
        # x values for detected line pixels
        self.allx = None
        # y values for detected line pixels
        self.ally = None

    def append_x_fitted(self, x_fitted):
        self.recent_x_fitted.append(x_fitted)
        self.allx = x_fitted

        if len(self.recent_x_fitted) > N_LINES:
            self.recent_x_fitted.pop(0)

        if len(self.recent_x_fitted) > 1:
            self.best_x = np.mean(self.recent_x_fitted, axis=0)
        else:
            self.best_x = x_fitted

    def append_fit(self, fit):
        self.recent_fit.append(fit)
        self.current_fit = fit

        if len(self.recent_fit) > N_LINES:
            self.recent_fit.pop(0)

        if len(self.recent_fit) > 1:
            self.best_fit = np.mean(self.recent_fit, axis=0)
        else:
            self.best_fit = fit

    def append_pos(self, pos):
        self.recent_pos.append(pos)
        self.line_base_pos = pos

        if len(self.recent_pos) > N_LINES:
            self.recent_pos.pop(0)

        if len(self.recent_pos) > 1:
            self.best_pos = np.average(self.line_base_pos)
        else:
            self.best_pos = pos

    def calculate_curvature(self):
        # Calculate the polynomial in real meters
        y_max = np.argmax(self.ally) * YM_PER_PIX
        fit_cr = np.polyfit(self.ally * YM_PER_PIX, self.best_x * XM_PER_PIX, 2)
        self.radius_of_curvature = ((1 + (2 * fit_cr[0] * y_max + fit_cr[1]) ** 2) ** 1.5) / np.absolute(2 * fit_cr[0])
        self.recent_radius.append(self.radius_of_curvature)

        if len(self.recent_radius) > N_LINES:
            self.recent_radius.pop(0)

        if len(self.recent_radius) > 1:
            self.best_radius = np.average(self.recent_radius)
        else:
            self.best_radius = self.radius_of_curvature

    def check_sanity_radius(self):
        return abs(self.radius_of_curvature - self.best_radius) < 200

    def check_sanity_pos(self):
        return abs(self.line_base_pos - self.best_pos) < 1
