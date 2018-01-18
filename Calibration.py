import os

import cv2
import numpy as np


def merge_captures(frame_L, frame_R):
    merged = []

    for rowL, rowR in zip(frame_L, frame_R):
        row = []

        row.extend(rowL)
        row.extend(rowR)

        merged.append(row)

    return np.array(merged)


def cheeseboard_capture(cameraL, cameraR):
    # termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objpL = np.zeros((7 * 7, 3), np.float32)
    objpL[:, :2] = np.mgrid[0:7, 0:7].T.reshape(-1, 2)

    objpR = np.zeros((7 * 7, 3), np.float32)
    objpR[:, :2] = np.mgrid[0:7, 0:7].T.reshape(-1, 2)

    # Arrays to store object points and image points from all the images.
    objpointsL = []  # 3d point in real world space
    objpointsR = []  # 3d point in real world space

    imgpointsL = []  # 2d points in image plane.
    imgpointsR = []  # 2d points in image plane.

    i = 0

    while True:
        rectL, imgL = cameraL.read()
        grayL = cv2.cvtColor(imgL, cv2.COLOR_BGR2GRAY)

        rectR, imgR = cameraR.read()
        grayR = cv2.cvtColor(imgR, cv2.COLOR_BGR2GRAY)

        # Find the chess board corners
        retL, cornersL = cv2.findChessboardCorners(grayL, (7, 7), None)
        retR, cornersR = cv2.findChessboardCorners(grayR, (7, 7), None)

        key_pressed = (cv2.waitKey(1) & 0xFF)

        if key_pressed == ord('q'):
            shapeL = grayL.shape[::-1]
            shapeR = grayR.shape[::-1]
            break

        # If found, add object points, image points (after refining them)
        if retL and retR:
            corners2L = cv2.cornerSubPix(grayL, cornersL, (11, 11), (-1, -1), criteria)
            corners2R = cv2.cornerSubPix(grayR, cornersR, (11, 11), (-1, -1), criteria)

            # Draw and display the corners
            imgL = cv2.drawChessboardCorners(imgL, (7, 7), corners2L, retL)
            imgR = cv2.drawChessboardCorners(imgR, (7, 7), corners2R, retR)

            if key_pressed == ord('c'):
                objpointsL.append(objpL)
                objpointsR.append(objpR)

                imgpointsL.append(corners2L)
                imgpointsR.append(corners2R)
                i += 1
                print(i)

        cv2.imshow('imgL', imgL)
        cv2.imshow('imgR', imgR)

    return objpointsL, imgpointsL, shapeL, objpointsR, imgpointsR, shapeR


def calibrate_camera(objpointsL, imgpointsL, shapeL, objpointsR, imgpointsR, shapeR):
    calibrated_cameraL = cv2.calibrateCamera(objpointsL, imgpointsL, shapeL, None, None)
    calibrated_cameraR = cv2.calibrateCamera(objpointsR, imgpointsR, shapeR, None, None)

    retL, mtxL, distL, rvecsL, tvecsL = calibrated_cameraL
    retR, mtxR, distR, rvecsR, tvecsR = calibrated_cameraR

    return [retL], mtxL, distL, rvecsL, tvecsL, [retR], mtxR, distR, rvecsR, tvecsR


def new_camera_matrix(shapeL, mtxL, distL, shapeR, mtxR, distR):
    wL, hL = shapeL
    newcameramtxL, roiL = cv2.getOptimalNewCameraMatrix(mtxL, distL, (wL, hL), 1, (wL, hL))

    wR, hR = shapeR
    newcameramtxR, roiR = cv2.getOptimalNewCameraMatrix(mtxR, distR, (wR, hR), 1, (wL, hL))

    return newcameramtxL, roiL, newcameramtxR, roiR


def undistort_frames(frameL, mtxL, distL, newcameramtxL, roiL, frameR, mtxR, distR, newcameramtxR, roiR):
    # undistort
    wL, hL = frameL.shape[:2]
    mapx, mapy = cv2.initUndistortRectifyMap(mtxL, distL, None, newcameramtxL, (wL, hL), 5)
    dstL = cv2.remap(frameL, mapx, mapy, cv2.INTER_LINEAR)

    wR, hR = frameR.shape[:2]
    mapx, mapy = cv2.initUndistortRectifyMap(mtxR, distR, None, newcameramtxR, (wR, hR), 5)
    dstR = cv2.remap(frameR, mapx, mapy, cv2.INTER_LINEAR)

    # # crop the image TODO make it so that it crops the same length
    # x, y, w, h = roiL
    # dstL = dstL[y:y + h, x:x + w]
    #
    # x, y, w, h = roiR
    # dstR = dstR[y:y + h, x:x + w]

    return dstL, dstR


def calibrate():
    cap_left = cv2.VideoCapture(2)
    cap_right = cv2.VideoCapture(1)

    undistorted_file_name = "right_left_undistorted.npy"
    calibrated_file_name = "right_left_calibrated.npy"
    new_camera_matrix_file_name = "right_left_new_camera_matrix.npy"

    if not os.path.exists(undistorted_file_name):
        right_left_undistorted = cheeseboard_capture(cap_left, cap_right)
        np.save(undistorted_file_name, right_left_undistorted)
    else:
        right_left_undistorted = np.load(undistorted_file_name)

    objL, imgL, shapeL, objR, imgR, shapeR = right_left_undistorted

    if not os.path.exists(calibrated_file_name):
        right_left_calibrated = calibrate_camera(objL, imgL, shapeL, objR, imgR, shapeR)

        np.save(calibrated_file_name, right_left_calibrated)
    else:
        right_left_calibrated = np.load(calibrated_file_name)

    retL, mtxL, distL, rvecsL, tvecsL, retR, mtxR, distR, rvecsR, tvecsR = right_left_calibrated
    retL = retL[0]
    retR = retR[0]

    if not os.path.exists(new_camera_matrix_file_name):
        right_left_new_camera_matrix = new_camera_matrix(shapeL, mtxL, distL, shapeR, mtxR, distR)
        np.save(new_camera_matrix_file_name, right_left_new_camera_matrix)
    else:
        right_left_new_camera_matrix = np.load(new_camera_matrix_file_name)

    newcameramtxL, roiL, newcameramtxR, roiR = right_left_new_camera_matrix

    cap_left.release()
    cap_right.release()

    mtx_L = mtxL
    dist_L = distL
    newcameramt_L = newcameramtxL
    roi_L = roiL

    mtx_R = mtxR
    dist_R = distR
    newcameramt_R = newcameramtxR
    roi_R = roiR

    cv2.destroyAllWindows()

calibrate()

mtx_L = None
dist_L = None
newcameramt_L = None
roi_L = None
frame_R = None
mtx_R = None
dist_R = None
newcameramtx_R = None
roi_R = None


def undistort(frameL, frameR):
    frameL, frameR = undistort_frames(frameL, mtx_L, dist_L, newcameramt_L, roi_L, frameR, mtx_R, dist_R,
                                      newcameramtx_R, roi_R)
    return frameL, frameR


# if __name__ == '__main__':
#     calibrate()
