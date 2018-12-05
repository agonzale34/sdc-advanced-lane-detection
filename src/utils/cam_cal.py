import numpy as np
import cv2
import glob
import pickle
from src.utils.params import *


# helper to calibrate a camera
def calibrate_camera(cols, rows, images_folder, images_regex, show_progress=False):

    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((cols * rows, 3), np.float32)
    objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)

    # Arrays to store object points and image points from all the images.
    objpoints = []  # 3d points in real world space
    imgpoints = []  # 2d points in image plane.

    # Make a list of calibration images
    images = glob.glob(images_folder + images_regex)

    img1 = cv2.imread(images[0])
    img_size = (img1.shape[1], img1.shape[0])

    # Step through the list and search for chessboard corners
    for idx, file_name in enumerate(images):
        img = cv2.imread(file_name)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Find the chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, (cols, rows), None)

        # If found, add object points, image points
        if ret:
            objpoints.append(objp)
            imgpoints.append(corners)

            # Draw and display the corners
            if show_progress:
                cv2.drawChessboardCorners(img, (cols, rows), corners, ret)
                cv2.imshow('img', img)
                cv2.waitKey(100)

    cv2.destroyAllWindows()

    # Do camera calibration given object points and image points
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_size, None, None)

    # Save the camera calibration result for later use (we won't worry about rvecs / tvecs)
    dist_pickle = {CC_MTX: mtx, CC_DIST: dist}
    pickle.dump(dist_pickle, open(images_folder + CC_FILE, "wb"))
    print("camera calibration saved successfully")

    # return the main values
    return mtx, dist


# helper to undistort images
def undistort_image(input_img, imtx, idist):
    return cv2.undistort(input_img, imtx, idist, None, imtx)
