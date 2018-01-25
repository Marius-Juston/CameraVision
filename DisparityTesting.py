import cv2

import Calibration
import Reconstruct

if __name__ == '__main__':

    cap_left = cv2.VideoCapture(1)
    cap_right = cv2.VideoCapture(0)

    Calibration.calibrate(cap_left, cap_right, clean_previous=False)

    rect, frameL = cap_left.read()
    h, w = frameL.shape[:2]
    stereo_bm = Reconstruct.StereoBM(w, h, .8 * w, show_settings=True)
    stereo_sgbm = Reconstruct.StereoSGBM(w, h, .8 * w, show_settings=True)

    while True:
        rect = cap_left.grab()
        rect = cap_right.grab()

        frameL = cap_left.retrieve()[1]
        frameR = cap_right.retrieve()[1]

        frameL, frameR = Calibration.undistort_rectify(frameL, frameR)

        cv2.imshow("L", frameL)
        cv2.imshow("R", frameR)

        stereo_bm.compute(frameL, frameR)
        stereo_sgbm.compute(frameL, frameR)

        key_press = cv2.waitKey(100) & 0xFF
        if key_press == ord('q'):
            break

    cap_left.release()
    cap_right.release()
    cv2.destroyAllWindows()
