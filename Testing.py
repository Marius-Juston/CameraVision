import cv2

import Calibration

if __name__ == '__main__':
    cap_left = cv2.VideoCapture(2)
    cap_right = cv2.VideoCapture(1)

    Calibration.calibrate(cap_left, cap_right)

    while True:
        rect, frameL = cap_left.read()
        rect, frameR = cap_right.read()

        key_press = cv2.waitKey(100) & 0xFF

        f1, f2 = Calibration.undistort(frameL, frameR)

        frameL, frameR = Calibration.undistory_rectify(frameL, frameR)

        cv2.imshow("L", frameL)
        cv2.imshow("R", frameR)

        if key_press == ord('q'):
            break

    cap_left.release()
    cap_right.release()
    cv2.destroyAllWindows()
