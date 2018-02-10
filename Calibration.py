# coding=utf-8
import os
import shutil
from datetime import datetime

import cv2
import numpy as np


def merge_captures(frame_l, frame_r):
    """

    :param frame_l:
    :param frame_r:
    :return:
    """
    merged = []

    for rowL, rowR in zip(frame_l, frame_r):
        row = []

        row.extend(rowL)
        row.extend(rowR)

        merged.append(row)

    return np.array(merged)


def cheeseboard_capture(camera_l, camera_r, number_columns=7, number_rows=7):
    """

    :param camera_l:
    :param camera_r:
    :param number_columns:
    :param number_rows:
    :return:
    """
    # termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp_l = np.zeros((number_rows * number_columns, 3), np.float32)
    objp_l[:, :2] = np.mgrid[0:number_rows, 0:number_columns].T.reshape(-1, 2)

    objp_r = np.zeros((number_rows * number_columns, 3), np.float32)
    objp_r[:, :2] = np.mgrid[0:number_rows, 0:number_columns].T.reshape(-1, 2)

    # Arrays to store object points and image points from all the images.
    objpoints_l = []  # 3d point in real world space
    objpoints_r = []  # 3d point in real world space

    imgpoints_l = []  # 2d points in image plane.
    imgpoints_r = []  # 2d points in image plane.

    i = 0

    while True:
        rect_l, img_l = camera_l.read()
        gray_l = cv2.cvtColor(img_l, cv2.COLOR_BGR2GRAY)

        rect_r, img_r = camera_r.read()
        gray_r = cv2.cvtColor(img_r, cv2.COLOR_BGR2GRAY)

        # Find the chess board corners
        ret_l, corners_l = cv2.findChessboardCorners(gray_l, (number_rows, number_columns))
        ret_r, corners_r = cv2.findChessboardCorners(gray_r, (number_rows, number_columns))

        key_pressed = (cv2.waitKey(1) & 0xFF)

        if key_pressed == ord('q'):
            shape_l = gray_l.shape[::-1]
            shape_r = gray_r.shape[::-1]
            break

        # If found, add object points, image points (after refining them)
        if ret_l and ret_r:
            corners2_l = cv2.cornerSubPix(gray_l, corners_l, (11, 11), (-1, -1), criteria)
            corners2_r = cv2.cornerSubPix(gray_r, corners_r, (11, 11), (-1, -1), criteria)

            # Draw and display the corners
            img_l = cv2.drawChessboardCorners(img_l, (7, 7), corners2_l, ret_l)
            img_r = cv2.drawChessboardCorners(img_r, (7, 7), corners2_r, ret_r)

            if key_pressed == ord('c'):
                objpoints_l.append(objp_l)
                objpoints_r.append(objp_r)

                imgpoints_l.append(corners2_l)
                imgpoints_r.append(corners2_r)
                i += 1
                print(i)

        cv2.imshow('imgL', img_l)
        cv2.imshow('imgR', img_r)

    return objpoints_l, imgpoints_l, shape_l, objpoints_r, imgpoints_r, shape_r


def calibrate_camera(objpoints_l, imgpoints_l, shape_l, objpoints_r, imgpoints_r, shape_r):
    """

    :param objpoints_l:
    :param imgpoints_l:
    :param shape_l:
    :param objpoints_r:
    :param imgpoints_r:
    :param shape_r:
    :return:
    """
    calibrated_camera_l = cv2.calibrateCamera(objpoints_l, imgpoints_l, shape_l, None, None)
    calibrated_camera_r = cv2.calibrateCamera(objpoints_r, imgpoints_r, shape_r, None, None)

    ret_l, mtx_l, dist_l, rvecs_l, tvecs_l = calibrated_camera_l
    ret_r, mtx_r, dist_r, rvecs_r, tvecs_r = calibrated_camera_r

    return [ret_l], mtx_l, dist_l, rvecs_l, tvecs_l, [ret_r], mtx_r, dist_r, rvecs_r, tvecs_r


def new_camera_matrix(shape_l, mtx_l, dist_l, shape_r, mtx_r, dist_r):
    """

    :param shape_l:
    :param mtx_l:
    :param dist_l:
    :param shape_r:
    :param mtx_r:
    :param dist_r:
    :return:
    """
    h_l, w_l = shape_l
    new_camera_mtx_l, roi_l = cv2.getOptimalNewCameraMatrix(mtx_l, dist_l, (w_l, h_l), 1, (w_l, h_l))

    h_r, w_r = shape_r
    new_camera_mtx_r, roi_r = cv2.getOptimalNewCameraMatrix(mtx_r, dist_r, (w_r, h_r), 1, (w_l, h_l))

    return new_camera_mtx_l, roi_l, new_camera_mtx_r, roi_r


def undistort_frames(frame_l, mtx_l, dist_l, new_camera_mtx_l, roi_l, frame_r, mtx_r, dist_r, new_camera_mtx_r, roi_r):
    """

    :param frame_l:
    :param mtx_l:
    :param dist_l:
    :param new_camera_mtx_l:
    :param roi_l:
    :param frame_r:
    :param mtx_r:
    :param dist_r:
    :param new_camera_mtx_r:
    :param roi_r:
    :return:
    """
    # undistort
    h_l, w_l = frame_l.shape[:2]
    mapx, mapy = cv2.initUndistortRectifyMap(mtx_l, dist_l, None, new_camera_mtx_l, (w_l, h_l), 5)
    dst_l = cv2.remap(frame_l, mapx, mapy, cv2.INTER_LINEAR)

    h_r, w_r = frame_r.shape[:2]
    mapx, mapy = cv2.initUndistortRectifyMap(mtx_r, dist_r, None, new_camera_mtx_r, (w_r, h_r), 5)
    dst_r = cv2.remap(frame_r, mapx, mapy, cv2.INTER_LINEAR)

    # # crop the image TODO make it so that it crops the same length
    x, y, w, h = roi_l
    dst_l = dst_l[y:y + h, x:x + w]

    x, y, w, h = roi_r
    dst_r = dst_r[y:y + h, x:x + w]

    return dst_l, dst_r


def set_camera_settings(camera):
    camera.set(cv2.CAP_PROP_ISO_SPEED, 1000)
    camera.set(cv2.CAP_PROP_FRAME_COUNT, 32)
    camera.set(cv2.CAP_PROP_AUTO_EXPOSURE, False)
    camera.set(cv2.CAP_PROP_AUTOFOCUS, False)
    camera.set(cv2.CAP_PROP_BRIGHTNESS, 100)
    pass


def calibrate(cap_left, cap_right, clean_previous=False, close_cameras=False, number_columns=7, number_rows=7):
    """

    :param cap_left:
    :param cap_right:
    :param clean_previous:
    :param close_cameras:
    """
    now = datetime.now().now()
    timeout = 10  # seconds
    while not cap_right.isOpened() or not cap_left.isOpened():
        if (datetime.now() - now).seconds >= timeout:
            raise Exception("Cameras not connect")

    set_camera_settings(cap_left)
    set_camera_settings(cap_right)

    calibration_folder = "./calibration_save/"

    undistorted_file_name = calibration_folder + "right_left_undistorted.npy"
    calibrated_file_name = calibration_folder + "right_left_calibrated.npy"
    new_camera_matrix_file_name = calibration_folder + "right_left_new_camera_matrix.npy"
    stereo_calibration_file_name = calibration_folder + "stereo_calibration.npy"
    stereo_rectification_file_name = calibration_folder + "stereo_rectification.npy"

    if clean_previous and os.path.exists(calibration_folder):
        shutil.rmtree(calibration_folder)

    if not os.path.exists(calibration_folder):
        os.makedirs(calibration_folder)

    if not os.path.exists(undistorted_file_name):
        right_left_undistorted = cheeseboard_capture(cap_left, cap_right, number_rows=number_rows,
                                                     number_columns=number_columns)
        np.save(undistorted_file_name, right_left_undistorted)
    else:
        right_left_undistorted = np.load(undistorted_file_name)

    obj_l, img_l, shape_l, obj_r, img_r, shape_r = right_left_undistorted

    if not os.path.exists(calibrated_file_name):
        right_left_calibrated = calibrate_camera(obj_l, img_l, shape_l, obj_r, img_r, shape_r)

        np.save(calibrated_file_name, right_left_calibrated)
    else:
        right_left_calibrated = np.load(calibrated_file_name)

    ret_l, mtx_l, dist_l, rvecs_l, tvecs_l, ret_r, mtx_r, dist_r, rvecs_r, tvecs_r = right_left_calibrated
    ret_l = ret_l[0]
    ret_r = ret_r[0]

    if not os.path.exists(new_camera_matrix_file_name):
        right_left_new_camera_matrix = new_camera_matrix(shape_l, mtx_l, dist_l, shape_r, mtx_r, dist_r)
        np.save(new_camera_matrix_file_name, right_left_new_camera_matrix)
    else:
        right_left_new_camera_matrix = np.load(new_camera_matrix_file_name)

    new_camera_mtx_l, roi_l, new_camera_mtx_r, roi_r = right_left_new_camera_matrix

    ############################################################################################

    stereocalib_criteria = (cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, 100, 1e-5)
    stereocalib_flags = cv2.CALIB_FIX_ASPECT_RATIO | cv2.CALIB_ZERO_TANGENT_DIST | cv2.CALIB_SAME_FOCAL_LENGTH | cv2.CALIB_RATIONAL_MODEL | cv2.CALIB_FIX_K3 | cv2.CALIB_FIX_K4 | cv2.CALIB_FIX_K5

    if not os.path.exists(stereo_calibration_file_name):
        stereo_calibration = cv2.stereoCalibrate(obj_l,
                                                 img_l,
                                                 img_r,
                                                 new_camera_mtx_l,
                                                 dist_l,
                                                 new_camera_mtx_r,
                                                 dist_r,
                                                 (640, 480),
                                                 criteria=stereocalib_criteria,
                                                 flags=stereocalib_flags)

        retval, new_camera_mtx_l, dist_l, new_camera_mtx_r, dist_r, r, t, e, f = stereo_calibration
        retval = [retval]

        stereo_calibration = [retval, new_camera_mtx_l, dist_l, new_camera_mtx_r, dist_r, r, t, e, f]

        np.save(stereo_calibration_file_name, stereo_calibration)
    else:
        stereo_calibration = np.load(stereo_calibration_file_name)

    retval, new_camera_mtx_l, dist_l, new_camera_mtx_r, dist_r, r, t, e, f = stereo_calibration
    retval = retval[0]

    rectify_scale = 0  # 0=full crop, 1=no crop

    if not os.path.exists(stereo_rectification_file_name):
        stereo_rectification = cv2.stereoRectify(new_camera_mtx_l, dist_l, new_camera_mtx_r, dist_r,
                                                 (640, 480), r, t, alpha=rectify_scale)

        np.save(stereo_rectification_file_name, stereo_rectification)
    else:
        stereo_rectification = np.load(stereo_rectification_file_name)

    r1, r2, p1, p2, q, roi1, roi2 = stereo_rectification

    left_maps = cv2.initUndistortRectifyMap(new_camera_mtx_l, dist_l, r1, p1, (640, 480), cv2.CV_32FC1)
    right_maps = cv2.initUndistortRectifyMap(new_camera_mtx_r, dist_r, r2, p2, (640, 480), cv2.CV_32FC1)

    cv2.destroyAllWindows()

    if close_cameras:
        cap_left.release()
        cap_right.release()

    #############################################################################

    global mtx_L, dist_L, newcameramtx_L, roi_L, mtx_R, dist_R, newcameramtx_R, roi_R, rectify_left_map, rectify_right_map
    mtx_L = mtx_l
    dist_L = dist_l
    newcameramtx_L = new_camera_mtx_l
    roi_L = roi_l

    mtx_R = mtx_r
    dist_R = dist_r
    newcameramtx_R = new_camera_mtx_r
    roi_R = roi_r

    rectify_left_map = left_maps
    rectify_right_map = right_maps


rectify_left_map = None
rectify_right_map = None
mtx_L = None
dist_L = None
newcameramtx_L = None
roi_L = None
mtx_R = None
dist_R = None
newcameramtx_R = None
roi_R = None


def undistort(frame_l, frame_r):
    """

    :param frame_l:
    :param frame_r:
    :return:
    """
    frame_l, frame_r = undistort_frames(frame_l, mtx_L, dist_L, newcameramtx_L, roi_L, frame_r, mtx_R, dist_R,
                                        newcameramtx_R, roi_R)

    return frame_l, frame_r


def undistort_rectify(frame_l, frame_r):
    """

    :param frame_l:
    :param frame_r:
    :return:
    """
    frame_l, frame_r = undistort(frame_l, frame_r)

    frame_l = cv2.remap(frame_l, rectify_left_map[0], rectify_left_map[1],
                        cv2.INTER_LANCZOS4)  # or cv2.INTER_NEAREST default cv2.INTER_LANCZOS4
    frame_r = cv2.remap(frame_r, rectify_right_map[0], rectify_right_map[1],
                        cv2.INTER_LANCZOS4)  # or cv2.INTER_NEAREST default cv2.INTER_LANCZOS4

    return frame_l, frame_r


if __name__ == '__main__':
    calibrate(cv2.VideoCapture(1), cv2.VideoCapture(2), clean_previous=False, close_cameras=True)
