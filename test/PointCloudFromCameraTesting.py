# coding=utf-8
import cv2

import Calibration
import Reconstruct
from PointCloudViewer import VtkPointCloud

if __name__ == '__main__':

    cap_left = cv2.VideoCapture(0)
    cap_right = cv2.VideoCapture(2)

    Calibration.calibrate(cap_left, cap_right, clean_previous=False, close_cameras=False)

    rect, frameL = cap_left.read()
    h, w = frameL.shape[:2]
    stereo = Reconstruct.StereoBM(w, h, .8 * w, show_settings=True, show_disparity=True)
    # stereo = Reconstruct.StereoSGBM(w, h, .8 * w, show_settings=True, show_disparity=True)

    get = True
    p = True

    with VtkPointCloud()as pointCloud:
        while True:
            if get:
                rect = cap_left.grab()
                rect = cap_right.grab()

                frameL = cap_left.retrieve()[1]
                frameR = cap_right.retrieve()[1]

                frameL, frameR = Calibration.undistort_rectify(frameL, frameR)

                cv2.imshow("L", frameL)
                cv2.imshow("R", frameR)

            if p:
                disp = stereo.compute(frameL, frameR)
                points, colors = stereo.to_3d(disp, frameL)

                # print(points)

                pointCloud.clear_points()
                pointCloud.add_points(points, colors)

            key_press = cv2.waitKey(100) & 0xFF
            if key_press == ord('q'):
                break
            elif key_press == ord(' '):
                get = not get

            elif key_press == ord('t'):
                p = not p
        cap_left.release()
        cap_right.release()
        cv2.destroyAllWindows()
