import os.path
import cv2
import glob
import pyrealsense2 as rs
import time
from pathlib import Path
import numpy as np

def read_image_pipline():
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
    profile = pipeline.start(config)

    save_path = os.path.join(os.getcwd(), "../calib_images", time.strftime("%Y_%m_%d", time.localtime()))
    if not Path(save_path).exists():
        os.mkdir(save_path)
    saved_color_image = None
    saved_count = 0

    cv2.namedWindow("live", cv2.WINDOW_AUTOSIZE)
    cv2.namedWindow("save", cv2.WINDOW_AUTOSIZE)

    try:
        while True:
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            color_image = np.asarray(color_frame.get_data())
            #color_image = cv2.resize(color_image, (256,192))
            cv2.imshow("live", color_image)
            key = cv2.waitKey(30)

            if key == ord('s'):
                saved_color_image = color_image
                cv2.imwrite(os.path.join((save_path), "{}.png".format(saved_count)), saved_color_image)
                saved_count+=1
                cv2.imshow("save", saved_color_image)

            if key == ord('q'):
                cv2.destroyAllWindows()
                break
    finally:
        pipeline.stop()


def getCameraParameters(path):
    chess_w = 7
    chess_h = 7

    # termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((chess_w*chess_h, 3), np.float32)
    objp[:,:2] = np.mgrid[0:chess_w, 0:chess_h].T.reshape(-1,2)
    objp = objp*2 # 2mm
    # Arrays to store object points and image points from all the images.
    objpoints = [] # 3d point in real world space
    imgpoints = [] # 2d points in image plane.
    path_name = path + '*.png'
    files = glob.glob(path_name)
    for file in files:
        img = cv2.imread(file)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Find the chess board corners
        ret, corners = cv2.findChessboardCorners(gray, (chess_w, chess_h),None)

        # If found, add object points, image points (after refining them)
        if ret == True:
            objpoints.append(objp)

            corners2 = cv2.cornerSubPix(gray,corners, (9,9), (-1,-1), criteria)
            imgpoints.append(corners2)

            # Draw and display the corners
            img = cv2.drawChessboardCorners(img, (chess_w, chess_h), corners2,ret)
            cv2.imshow('img',img)
            cv2.waitKey(200)

    # camera matrix, distortion coefficients, rotation and translation vectors
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1],None,None)

    print("Number of valid views: " + str(len(rvecs)))
    print("Camera Intrinsics \n" + str(mtx))
    print("Camera Distortion Matrix \n" + str(dist))
    cv2.destroyAllWindows()
    return mtx, rvecs, tvecs

if __name__ == '__main__':
    read_image_pipline()
    getCameraParameters('../calib_images/2021_12_16/')