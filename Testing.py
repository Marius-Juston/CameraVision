import cv2

import Calibration

if __name__ == '__main__':
    Calibration.calibrate()

    cap_left = cv2.VideoCapture(2)
    cap_right = cv2.VideoCapture(1)

    while True:
        pass
