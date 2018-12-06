from src.utils.cam_cal import *
from src.utils.plt_utils import *
from src.utils.image_utils import *
from src.utils.params import *
from src.models.settings import *
import matplotlib.image as mpimg


def process_image(p_image, settings: Settings):
    """
    Pipeline with the complete image process
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
    un_dist = undistort_image(p_image, settings.mtx, settings.dist)

    # Apply the advanced transform to get the final image
    img_adv = advanced_transform(
        un_dist, kernel_size=KERNEL_SIZE, s_thresh=S_CHANNEL_THRESH, sx_thresh=SOBEL_X_THRESH
    )

    # Get the bird eye view
    img_bird_eye = bird_eye_transform(img_adv, settings.aoi_src, settings.bird_dst)

    visualize_result(p_image, img_bird_eye, gray=True)


# testing the pipeline
cam_cal_path = '../resources/camera_cal/'
# 1. calibrate the camera
# mtx, dist = calibrate_camera(9, 6, cam_cal_path, 'cali*.jpg')
# Read in the saved mtx and dist
pickle = pickle.load(open(cam_cal_path + CC_FILE, 'rb'))

settings_vid1 = Settings()

settings_vid1.mtx = pickle[CC_MTX]
settings_vid1.dist = pickle[CC_DIST]
settings_vid1.aoi_xmid = 0.511
settings_vid1.aoi_ymid = 0.625
settings_vid1.aoi_upsz = 0.08
settings_vid1.aoi_upds = 15
settings_vid1.aoi_basesz = 0.390

image = mpimg.imread('../resources/test_images/test2.jpg')
# image = mpimg.imread('../resources/test_images/straight_lines2.jpg')
settings_vid1.find_aoi_src_dst(image)
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

process_image(image, settings_vid1)
