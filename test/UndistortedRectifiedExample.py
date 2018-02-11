# coding=utf-8
import cv2

import Calibration

if __name__ == '__main__':

    cap_left = cv2.VideoCapture(0)
    cap_right = cv2.VideoCapture(2)

    Calibration.calibrate(cap_left, cap_right, clean_previous=False, close_cameras=False)

    rect, frameL = cap_left.read()
    h, w = frameL.shape[:2]

    while True:
        rect = cap_left.grab()
        rect = cap_right.grab()

        frameL = cap_left.retrieve()[1]
        frameR = cap_right.retrieve()[1]

        frameL, frameR = Calibration.undistort_rectify(frameL, frameR)

        cv2.imshow("L", frameL)
        cv2.imshow("R", frameR)

        key_press = cv2.waitKey(100) & 0xFF
        if key_press == ord('q'):
            break

    cap_left.release()
    cap_right.release()
    cv2.destroyAllWindows()
