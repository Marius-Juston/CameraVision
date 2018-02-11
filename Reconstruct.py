# coding=utf-8
import math
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
    """

    :param fn:
    :param verts:
    :param colors:
    """
    verts = verts.reshape(-1, 3)
    colors = colors.reshape(-1, 3)
    verts = np.hstack([verts, colors])
    with open(fn, 'wb') as f:
        f.write((ply_header % dict(vert_num=len(verts))).encode('utf-8'))
        np.savetxt(f, verts, fmt='%f %f %f %d %d %d ')


class Stereo(object):
    """

    """

    def __init__(self, w, h, focal_length, window_name="window", show_settings=False,
                 show_disparity=True):
        self.show_disparity = show_disparity
        self.window_name = window_name
        self.Q = np.float32([[1, 0, 0, -0.5 * w],
                             [0, -1, 0, 0.5 * h],  # turn points 180 deg around x-axis,
                             [0, 0, 0, -focal_length],  # so that y-axis looks up
                             [0, 0, 1, 0]])

        self.filter = cv2.ximgproc_DisparityWLSFilter()

        if show_disparity:
            cv2.namedWindow(self.window_name)

        if show_settings:
            self.settings_name = window_name + " Settings"
            cv2.namedWindow(self.settings_name)

    def to_3d(self, disp, color_frame_l, distance_threshold=10, invalid=0):
        """

        :param distance_threshold:
        :param disp:
        :param color_frame_l:
        :return:
        """
        points = cv2.reprojectImageTo3D(disp, self.Q)
        # points[::, 2] *= -1

        mask = disp > disp.min()

        # print(points.shape)
        x = points[:, :, 0]  # TODO just get the x depth
        y = points[:, :, 1]
        z = points[:, :, 2]

        x[np.isnan(x)] = invalid
        y[np.isnan(y)] = invalid
        z[np.isnan(z)] = invalid

        x[np.isinf(x)] = invalid
        y[np.isinf(y)] = invalid
        z[np.isinf(z)] = invalid
        #
        # x_infinit = np.isinf(x)
        # y_infinit = np.isinf(x)
        # z_infinit = np.isinf(x)
        #
        #
        #
        # points_mask = np.logical_or(x_mask, y_mask, z_mask)
        # points_mask = np.logical_or(x_infinit, y_infinit, points_mask)
        # points_mask = np.logical_or(points_mask, z_infinit)
        #
        # x[points_mask] = -16
        # y[points_mask] = -16
        # z[points_mask] = -16
        # # print(x.shape)
        # # print(y.shape)
        # # print(z.shape)

        # print('x', x.max(), x.min())
        # print('y', y.max(), y.min())
        # print('z', z.max(), z.min())
        distance = np.sqrt(x ** 2 + y ** 2 + z ** 2)

        # print(mask)
        distance_mask = distance > distance_threshold

        # print(distance_mask)

        full_mask = np.logical_or(mask, distance_mask)

        out_points = points[full_mask]
        out_colors = color_frame_l[full_mask]

        return out_points, out_colors

    @abstractmethod
    def __change(self, x):
        pass

    @abstractmethod
    def compute(self, frame_l, frame_r):
        """

        :param frame_l:
        :param frame_r:
        """
        pass


class StereoBM(Stereo):
    """

    """

    def __init__(self, w, h, focal_length, show_settings=True, show_disparity=False, num_disparities=16, block_size=5):
        super(StereoBM, self).__init__(h, w, focal_length, "StereoBM disparity", show_settings, show_disparity)
        self.stereo = cv2.StereoBM_create(num_disparities, block_size)  # best between the options 0,5 or 16,5

        if show_settings:
            cv2.createTrackbar("numDisparities", self.settings_name, int(num_disparities / 16), 20, self.__change)
            cv2.setTrackbarMin("numDisparities", self.settings_name, 1)
            cv2.setTrackbarMax("numDisparities", self.settings_name, int(math.floor(min(w, h) / 16)))

            cv2.createTrackbar("blockSize", self.settings_name, block_size, 255, self.__change)
            cv2.setTrackbarMin("blockSize", self.settings_name, 5)

    def __change(self, x):
        block_size = cv2.getTrackbarPos("blockSize", self.settings_name)
        if block_size % 2 == 0:
            block_size += 1
            cv2.setTrackbarPos("blockSize", self.settings_name, block_size)

        num_disparities = max(cv2.getTrackbarPos("numDisparities", self.settings_name) * 16, 16)

        # print(block_size)

        self.stereo.setNumDisparities(num_disparities)
        self.stereo.setBlockSize(block_size)

    def compute(self, frame_l, frame_r):
        """

        :param frame_l:
        :param frame_r:
        :return:
        """
        frame_l = cv2.cvtColor(frame_l, cv2.COLOR_BGR2GRAY)
        frame_r = cv2.cvtColor(frame_r, cv2.COLOR_BGR2GRAY)

        # cv2.imshow("left", frame_l)
        # cv2.imshow("right", frame_r)

        disp = self.stereo.compute(frame_l, frame_r)
        # print(np.histogram(disp, 10))

        self.filter.filter(disp, frame_l)

        if self.show_disparity:
            disparity_visual = np.zeros(frame_l.shape, dtype=np.uint8)

            cv2.normalize(disp, disparity_visual, 0, 255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            disparity_visual = np.array(disparity_visual)

            # print(disparity_visual)

            # norm_coeff = 255 / disp.max()  # TODO make this work it is not showing anything
            # cv2.imshow(self.window_name, disp * norm_coeff / 255 )

            cv2.resize(disparity_visual, (720, 480), disparity_visual)
            cv2.imshow(self.window_name, disparity_visual)

        return disp


class StereoSGBM(Stereo):
    """

    """

    def compute(self, frame_l, frame_r):
        """

        :param frame_l:
        :param frame_r:
        :return:
        """
        disp = self.stereo.compute(frame_l, frame_r).astype(np.float32) / 16.0
        min_disparity = self.stereo.getMinDisparity()
        num_disparity = self.stereo.getNumDisparities()
        disparity = (disp - min_disparity) / num_disparity

        if self.show_disparity:
            cv2.resize(disparity, (720, 480), disparity)
            cv2.imshow(self.window_name, disparity)
        return disp

    def __init__(self, w, h, focal_length, show_settings=True, show_disparity=False,
                 default_window_size=3,
                 default_min_disp=16,
                 default_num_disp=16,
                 default_uniqueness_ratio=10,
                 default_speckle_window_size=100,
                 default_speckle_range=32,
                 default_disp12_max_diff=1):

        super(StereoSGBM, self).__init__(w, h, focal_length, "StereoSGBM disparity", show_settings, show_disparity)

        self.stereo = cv2.StereoSGBM_create(minDisparity=default_min_disp,
                                            numDisparities=default_num_disp,
                                            uniquenessRatio=default_uniqueness_ratio,
                                            speckleWindowSize=default_speckle_window_size,
                                            speckleRange=default_speckle_range,
                                            disp12MaxDiff=default_disp12_max_diff,
                                            P1=8 * 3 * default_window_size ** 2,
                                            P2=32 * 3 * default_window_size ** 2,
                                            )

        if show_settings:
            cv2.createTrackbar("minDisparity", self.settings_name, int(default_min_disp / 16), 30, self.__change)
            cv2.setTrackbarMin("minDisparity", self.settings_name, 1)

            cv2.createTrackbar("numDisparities", self.settings_name, int(default_num_disp / 16), 30, self.__change)
            cv2.setTrackbarMin("numDisparities", self.settings_name, 1)
            cv2.createTrackbar("uniquenessRatio", self.settings_name, 10, 40, self.__change)
            cv2.setTrackbarMin("uniquenessRatio", self.settings_name, 1)

            cv2.createTrackbar("speckleWindowSize", self.settings_name, 100, 200, self.__change)
            cv2.createTrackbar("speckleRange", self.settings_name, 32, 100, self.__change)
            cv2.createTrackbar("disp12MaxDiff", self.settings_name, 1, 20, self.__change)

            cv2.createTrackbar("window_size", self.settings_name, default_window_size, 10, self.__change)

    def __change(self, x):
        min_disparity = cv2.getTrackbarPos("minDisparity", self.settings_name) * 16
        num_disparity = cv2.getTrackbarPos("numDisparities", self.settings_name) * 16
        uniqueness_ratio = cv2.getTrackbarPos("uniquenessRatio", self.settings_name)
        speckle_window_size = cv2.getTrackbarPos("speckleWindowSize", self.settings_name)
        speckle_range = cv2.getTrackbarPos("speckleRange", self.settings_name)
        disp12_max_diff = cv2.getTrackbarPos("disp12MaxDiff", self.settings_name)
        window_size = cv2.getTrackbarPos("window_size", self.settings_name)

        self.stereo.setMinDisparity(min_disparity)
        self.stereo.setNumDisparities(num_disparity)
        self.stereo.setUniquenessRatio(uniqueness_ratio)
        self.stereo.setSpeckleWindowSize(speckle_window_size)
        self.stereo.setSpeckleRange(speckle_range)
        self.stereo.setDisp12MaxDiff(disp12_max_diff)
        self.stereo.setP1(8 * 3 * window_size ** 2)
        self.stereo.setP2(32 * 3 * window_size ** 2)
