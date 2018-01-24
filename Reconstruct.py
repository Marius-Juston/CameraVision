from abc import abstractmethod

import cv2
import numpy as np

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


class Stereo:
    def __init__(self, w, h, focal_length, window_name="window", show_settings=True):
        self.window_name = window_name
        self.Q = np.float32([[1, 0, 0, -0.5 * w],
                             [0, -1, 0, 0.5 * h],  # turn points 180 deg around x-axis,
                             [0, 0, 0, -focal_length],  # so that y-axis looks up
                             [0, 0, 1, 0]])

        cv2.namedWindow(self.window_name)

        if show_settings:
            self.settings_name = window_name + " Settings"
            cv2.namedWindow(self.settings_name)

    @abstractmethod
    def _to_3d(self, disp, color_frame_l):
        points = cv2.reprojectImageTo3D(disp, self.Q)
        return points

    @abstractmethod
    def __change(self, x):
        pass

    @abstractmethod
    def compute(self, frame_l, frame_r):
        pass


class StereoBM(object, Stereo):

    def _to_3d(self, disp, color_frame_l):
        points = super(StereoBM, self)._to_3d(disp, color_frame_l)
        mask = disp > disp.min()

        out_points = points[mask]
        out_colors = color_frame_l[mask]

        # TODO

    def __init__(self, w, h, focal_length, show_settings=True):
        super(object, self).__init__(h, w, focal_length, "StereoBM disparity", show_settings)

        self.stereo = cv2.StereoBM_create(16, 5)  # best between the options 0,5 or 16,5

        if show_settings:
            cv2.createTrackbar("numDisparities", self.settings_name, 1, 20, self.__change)
            cv2.setTrackbarMin("numDisparities", self.settings_name, 1)
            cv2.createTrackbar("blockSize", self.settings_name, 5, 255, self.__change)
            cv2.setTrackbarMin("blockSize", self.settings_name, 5)

    def __change(self, x):
        block_size = cv2.getTrackbarPos("blockSize", self.window_name)
        if block_size % 2 == 0:
            block_size += 1
            cv2.setTrackbarPos("blockSize", self.window_name, block_size)

        num_disparities = max(cv2.getTrackbarPos("numDisparities", self.settings_name) * 16, 16)

        self.stereo.setNumDisparities(num_disparities)
        self.stereo.setBlockSize(block_size)

    def compute(self, frame_l, frame_r):
        frame_l = cv2.cvtColor(frame_l, cv2.COLOR_BGR2GRAY)
        frame_r = cv2.cvtColor(frame_r, cv2.COLOR_BGR2GRAY)

        disp = self.stereo.compute(frame_l, frame_r)

        norm_coeff = 255 / disp.max()
        cv2.imshow(self.window_name, disp * norm_coeff / 255)

        return disp


class StereoSGBM(object, Stereo):
    def compute(self, frame_l, frame_r):
        disp = self.stereo.compute(frame_l, frame_r).astype(np.float32) / 16.0

        min_disparity = self.stereo.getMinDisparity()
        numDisparity = self.stereo.getNumDisparities()

        disparity = (disp - min_disparity) / numDisparity
        cv2.imshow(self.window_name, disparity)

        return disp

    def __init__(self, w, h, focal_length, show_settings=True):
        super(object, self).__init__(w, h, focal_length, "StereoSGBM disparity", show_settings)

        window_size = 3
        min_disp = 16
        num_disp = 112 - min_disp
        self.stereo = cv2.StereoSGBM_create(minDisparity=min_disp,
                                            numDisparities=num_disp,
                                            uniquenessRatio=10,
                                            speckleWindowSize=100,
                                            speckleRange=32,
                                            disp12MaxDiff=1,
                                            P1=8 * 3 * window_size ** 2,
                                            P2=32 * 3 * window_size ** 2,
                                            )

        if show_settings:
            cv2.createTrackbar("minDisparity", self.settings_name, int(min_disp / 16), 30, self.__change)
            cv2.setTrackbarMin("minDisparity", self.settings_name, 1)

            cv2.createTrackbar("numDisparities", self.settings_name, int(num_disp / 16), 30, self.__change)
            cv2.setTrackbarMin("numDisparities", self.settings_name, 1)
            cv2.createTrackbar("uniquenessRatio", self.settings_name, 10, 40, self.__change)
            cv2.setTrackbarMin("uniquenessRatio", self.settings_name, 1)
            #
            cv2.createTrackbar("speckleWindowSize", self.settings_name, 100, 200, self.__change)
            cv2.createTrackbar("speckleRange", self.settings_name, 32, 100, self.__change)
            cv2.createTrackbar("disp12MaxDiff", self.settings_name, 1, 20, self.__change)
            #
            cv2.createTrackbar("window_size", self.settings_name, window_size, 10, self.__change)

    def _to_3d(self, disp, color_frame_l):
        pass  # TODO

    def __change(self, x):
        min_disparity = cv2.getTrackbarPos("minDisparity", self.settings_name) * 16
        numDisparity = cv2.getTrackbarPos("numDisparities", self.settings_name) * 16
        uniquenessRatio = cv2.getTrackbarPos("uniquenessRatio", self.settings_name)
        speckleWindowSize = cv2.getTrackbarPos("speckleWindowSize", self.settings_name)
        speckleRange = cv2.getTrackbarPos("speckleRange", self.settings_name)
        disp12MaxDiff = cv2.getTrackbarPos("disp12MaxDiff", self.settings_name)
        window_size = cv2.getTrackbarPos("window_size", self.settings_name)

        self.stereo.setMinDisparity(min_disparity)
        self.stereo.setNumDisparities(numDisparity)
        self.stereo.setUniquenessRatio(uniquenessRatio)
        self.stereo.setSpeckleWindowSize(speckleWindowSize)
        self.stereo.setSpeckleRange(speckleRange)
        self.stereo.setDisp12MaxDiff(disp12MaxDiff)
        self.stereo.setP1(8 * 3 * window_size ** 2)
        self.stereo.setP2(32 * 3 * window_size ** 2)
