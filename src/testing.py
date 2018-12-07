from src.utils.cam_cal import *
from src.utils.lane_lines import *
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

image = mpimg.imread('../resources/test_images/test6.jpg')
# image = mpimg.imread('../resources/test_images/straight_lines2.jpg')
settings.find_aoi_src_dst(image)

un_dist = undistort_image(image, settings.mtx, settings.dist)

# Apply the advanced transform to get the final image
img_adv = advanced_transform(
    un_dist, kernel_size=KERNEL_SIZE, s_thresh=S_CHANNEL_THRESH, sx_thresh=SOBEL_X_THRESH
)

# Get the bird eye view
binary_warped, minv = bird_eye_transform(img_adv, settings.aoi_src, settings.bird_dst)

left_lane, right_line, offset = find_lane_lines_sliding_windows(np.copy(binary_warped))

result = draw_final_lines(binary_warped, minv, un_dist, left_lane, right_line, offset)

visualize_result(image, result)

# image = mpimg.imread('../resources/test_images/test2.jpg')
# # image = mpimg.imread('../resources/test_images/straight_lines2.jpg')
# settings_vid1.find_aoi_src_dst(image)
# un_dist = undistort_image(image, settings_vid1.mtx, settings_vid1.dist)
# img_aoi_src = np.copy(un_dist)
# img_aoi = np.copy(un_dist)
# cv2.polylines(img_aoi_src, [settings_vid1.aoi_src.astype(int)], True, color=(0, 255, 255), thickness=3)
# img_bird_eye = bird_eye_transform(img_aoi, settings_vid1.aoi_src, settings_vid1.bird_dst)
# cv2.polylines(img_bird_eye, [settings_vid1.bird_dst.astype(int)], True, color=(0, 255, 255), thickness=3)
# img_adv = advanced_transform(
#     un_dist, kernel_size=KERNEL_SIZE, s_thresh=S_CHANNEL_THRESH, sx_thresh=SOBEL_X_THRESH
# )
# visualize_result4(
#     (image, 'Original'), (un_dist, 'Undistort'), (img_aoi_src, 'Interest area'), (img_bird_eye, 'Bird Eye')
# )
