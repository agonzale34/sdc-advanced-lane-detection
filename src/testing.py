import matplotlib.image as mpimg

from src.lane_lines_detection import *
from src.utils.plt_utils import *

# testing the pipeline
cam_cal_path = '../resources/camera_cal/'
# 1. calibrate the camera
# mtx, dist = calibrate_camera(9, 6, cam_cal_path, 'cali*.jpg')
# Read in the saved mtx and dist
pickle = pickle.load(open(cam_cal_path + CC_FILE, 'rb'))

settings.mtx = pickle[CC_MTX]
settings.dist = pickle[CC_DIST]
settings.aoi_xmid = 0.511
settings.aoi_ymid = 0.63
settings.aoi_upsz = 0.015
settings.aoi_des = -35
settings.aoi_basesz = 0.350

image = mpimg.imread('../resources/test_images/test3.jpg')
# image = mpimg.imread('../resources/test_images/straight_lines2.jpg')
settings.find_aoi_src_dst(image)
result = process_image(image)
visualize_result(image, result)
