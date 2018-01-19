import cv2
import numpy as np

import Calibration

if __name__ == '__main__':

    cap_left = cv2.VideoCapture(0)
    cap_right = cv2.VideoCapture(1)

    Calibration.calibrate(cap_left, cap_right, clean_previous=False)

    min_disp = 16
    num_disp = 112 - min_disp
    stereo = cv2.StereoSGBM_create(num_disp, 16)

    while True:
        rect, frameL = cap_left.read()
        rect, frameR = cap_right.read()

        key_press = cv2.waitKey(100) & 0xFF

        # f1, f2 = Calibration.undistort(frameL, frameR)
        #
        # cv2.imshow("L undist", f1)
        # cv2.imshow("R undist", f2)

        frameL, frameR = Calibration.undistort_rectify(frameL, frameR)

        cv2.imshow("L", frameL)
        cv2.imshow("R", frameR)

        frameL = cv2.cvtColor(frameL, cv2.COLOR_BGR2GRAY)
        frameR = cv2.cvtColor(frameR, cv2.COLOR_BGR2GRAY)

        disp = stereo.compute(frameL, frameR)
        norm_coeff = 255 / disp.max()
        cv2.imshow("disparity", disp * norm_coeff / 255)

        h, w = frameL.shape[:2]
        f = 0.8 * w  # guess for focal length
        Q = np.float32([[1, 0, 0, -0.5 * w],
                        [0, -1, 0, 0.5 * h],  # turn points 180 deg around x-axis,
                        [0, 0, 0, -f],  # so that y-axis looks up
                        [0, 0, 1, 0]])
        points = cv2.reprojectImageTo3D(disp, Q)

        if key_press == ord('q'):
            break

    cap_left.release()
    cap_right.release()
    cv2.destroyAllWindows()
