from src.models.settings import Settings
"""
This files contains the constants for all the process
"""
CC_MTX = 'mtx'
CC_DIST = 'dist'
CC_FILE = 'camera_cali_pickle.p'

KERNEL_SIZE = 3
S_CHANNEL_THRESH = (120, 255)
SOBEL_X_THRESH = (20, 100)

# Choose the number of sliding windows
SW_NWINDOWS = 9
# Set the width of the windows +/- margin
SW_MARGIN = 200
# Set minimum number of pixels found to recenter window
SW_MIN_PIX = 50

# Define conversions in x and y from pixels space to meters
YM_PER_PIX = 30 / 720  # meters per pixel in y dimension
XM_PER_PIX = 3.7 / 700  # meters per pixel in x dimension

settings = Settings()
