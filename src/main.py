import matplotlib.image as mpimg
from moviepy.editor import VideoFileClip

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
    left_lane, right_line, offset = find_lane_lines_sliding_windows(np.copy(binary_warped))
    # 5. Draw the lines in the un-distorted image
    result = draw_final_lines(binary_warped, minv, un_dist, left_lane, right_line, offset)
    return result


# Calibrating the camera
cam_cal_path = '../resources/camera_cal/'
# Run the next command the first time
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

image = mpimg.imread('../resources/test_images/test2.jpg')
# # image = mpimg.imread('../resources/test_images/straight_lines2.jpg')
settings.find_aoi_src_dst(image)
process_image(image)

# white_output = '../resources/test_videos/project_video_out.mp4'
white_output = '../resources/test_videos/harder_challenge_video_out.mp4'
# To speed up the testing process you may want to try your pipeline on a shorter subclip of the video
# To do so add .subclip(start_second,end_second) to the end of the line below
# Where start_second and end_second are integer values representing the start and end of the subclip
# You may also uncomment the following line for a subclip of the first 5 seconds
# clip1 = VideoFileClip("../resources/test_videos/project_video.mp4").subclip(0,5)
clip1 = VideoFileClip("../resources/test_videos/harder_challenge_video.mp4").subclip(0,5)
# clip1 = VideoFileClip("test_videos/solidWhiteRight.mp4")
white_clip = clip1.fl_image(process_image)  # NOTE: this function expects color images!!
white_clip.write_videofile(white_output, audio=False)
