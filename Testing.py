import cv2
import numpy as np

import Calibration

ply_header = '''ply
format ascii 1.0
element vertex %(vert_num)d
property float x
property float y
property float z
property uchar red
property uchar green
property uchar blue
end_header
'''


def write_ply(fn, verts, colors):
    verts = verts.reshape(-1, 3)
    colors = colors.reshape(-1, 3)
    verts = np.hstack([verts, colors])
    with open(fn, 'wb') as f:
        f.write((ply_header % dict(vert_num=len(verts))).encode('utf-8'))
        np.savetxt(f, verts, fmt='%f %f %f %d %d %d ')


if __name__ == '__main__':

    cap_left = cv2.VideoCapture(0)
    cap_right = cv2.VideoCapture(1)

    Calibration.calibrate(cap_left, cap_right, clean_previous=False)

    window_size = 5
    min_disp = 80
    num_disp = 160 - min_disp
    stereo = cv2.StereoSGBM_create(minDisparity=min_disp,
                            numDisparities=num_disp,
                            uniquenessRatio=0,
                            speckleWindowSize=10,
                            speckleRange=10,
                            disp12MaxDiff=1,
                            P1=8 * 3 * window_size ** 2,
                            P2=32 * 3 * window_size ** 2,
                            )

    while True:
        rect, frameL = cap_left.read()
        rect, frameR = cap_right.read()

        key_press = cv2.waitKey(100) & 0xFF

        # f1, f2 = Calibration.undistort(frameL, frameR)
        #
        # cv2.imshow("L undist", f1)
        # cv2.imshow("R undist", f2)

        frameL, frameR = Calibration.undistort_rectify(frameL, frameR)

        # cv2.imshow("L", frameL)
        cv2.imshow("R", frameR)

        # frameL = cv2.cvtColor(frameL, cv2.COLOR_BGR2GRAY)
        # frameR = cv2.cvtColor(frameR, cv2.COLOR_BGR2GRAY)

        # disp = stereo.compute(frameL, frameR)
        disp = stereo.compute(frameL, frameR).astype(np.float32) / 16.0

        # norm_coeff = 255 / disp.max()
        # cv2.imshow("disparity", disp * norm_coeff / 255)

        h, w = frameL.shape[:2]
        f = 0.8 * w  # guess for focal length
        Q = np.float32([[1, 0, 0, -0.5 * w],
                        [0, -1, 0, 0.5 * h],  # turn points 180 deg around x-axis,
                        [0, 0, 0, -f],  # so that y-axis looks up
                        [0, 0, 1, 0]])

        points = cv2.reprojectImageTo3D(disp, Q)
        colors = cv2.cvtColor(frameL, cv2.COLOR_BGR2RGB)
        mask = disp > disp.min()

        cv2.imshow('left', frameL)
        disparity = (disp - min_disp) / num_disp
        cv2.imshow('disparity', disparity)

        out_points = points[mask]
        out_colors = colors[mask]

        out_fn = 'out.ply'

        if key_press == ord('q'):
            write_ply('out.ply', out_points, out_colors)
            break
        # break

    cap_left.release()
    cap_right.release()
    cv2.destroyAllWindows()
