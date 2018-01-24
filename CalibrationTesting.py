import cv2

import Calibration

if __name__ == '__main__':
    cap_left = cv2.VideoCapture(1)
    cap_right = cv2.VideoCapture(0)

    Calibration.calibrate(cap_left, cap_right, clean_previous=False)

    cap_left.release()
    cap_right.release()
    cv2.destroyAllWindows()
