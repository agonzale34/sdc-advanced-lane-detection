from src.utils.cam_cal import *
from src.utils.plt_utils import *
from src.utils.image_utils import *
from src.utils.params import *
import matplotlib.image as mpimg

"""
The goals / steps of this project are the following:
* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.
"""

cam_cal_path = '../resources/camera_cal/'

# 1. calibrate the camera
# mtx, dist = calibrate_camera(9, 6, cam_cal_path, 'cali*.jpg')
# Read in the saved mtx and dist
pickle = pickle.load(open(cam_cal_path + CC_FILE, 'rb'))
mtx = pickle[CC_MTX]
dist = pickle[CC_DIST]


# Pipeline with the complete image process
def process_image(p_image, p_mtx, p_dist, p_src, p_dst):
    un_dist = undistort_image(p_image, p_mtx, p_dist)

    # Apply the advanced transform to get the final image
    img_adv = advanced_transform(
        un_dist, kernel_size=KERNEL_SIZE, s_thresh=S_CHANNEL_THRESH, sx_thresh=SOBEL_X_THRESH
    )

    # Get the bird eye view
    img_bird_eye, M = bird_eye_transform(img_adv, p_src, p_dst)

    visualize_result(p_image, img_bird_eye, gray=True)


# image = mpimg.imread('../resources/test_images/test6.jpg')
image = mpimg.imread('../resources/test_images/straight_lines2.jpg')
src, dst = get_aoi_src_dst(image)
img_aoi_src = np.copy(image)
img_aoi = np.copy(image)
# Get the bird eye view
cv2.polylines(img_aoi_src, [src.astype(int)], True, color=(0, 255, 255), thickness=2)
img_bird_eye, M = bird_eye_transform(img_aoi, src, dst)
cv2.polylines(img_bird_eye, [dst.astype(int)], True, color=(0, 255, 255), thickness=2)
visualize_result4(image, img_aoi_src, img_aoi_src, img_bird_eye)

# process_image(image, mtx, dist, src, dst)
