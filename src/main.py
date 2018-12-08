import matplotlib.image as mpimg
from moviepy.editor import VideoFileClip

from src.lane_lines_detection import *


def process_video(video, video_out, total=True):
    if total:
        clip = VideoFileClip(video)
    else:
        clip = VideoFileClip(video).subclip(0, 7)
    white_clip = clip.fl_image(process_image)  # NOTE: this function expects color images!!
    white_clip.write_videofile(video_out, audio=False)


# Calibrating the camera
cam_cal_path = '../resources/camera_cal/'
# Run the next command the first time
# mtx, dist = calibrate_camera(9, 6, cam_cal_path, 'cali*.jpg')
# Read in the saved mtx and dist
pickle = pickle.load(open(cam_cal_path + CC_FILE, 'rb'))

settings.mtx = pickle[CC_MTX]
settings.dist = pickle[CC_DIST]
settings.aoi_xmid = 0.511
settings.aoi_ymid = 0.65
settings.aoi_upsz = 0.08
settings.aoi_upds = 15
settings.aoi_basesz = 0.390

image = mpimg.imread('../resources/test_images/test2.jpg')
# # image = mpimg.imread('../resources/test_images/straight_lines2.jpg')
settings.find_aoi_src_dst(image)

process_video(
  '../resources/test_videos/project_video.mp4', '../resources/output_videos/project_video.mp4', total=True
)
# process_video(
#     '../resources/test_videos/challenge_video.mp4', '../resources/output_videos/challenge_video.mp4', total=False
# )
