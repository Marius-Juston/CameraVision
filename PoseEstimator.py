# coding=utf-8
from abc import abstractmethod

import cv2


class Estimator(object):
    """

    """

    def correct_3d_point_position(self, points):
        """

        :param points:
        """
        pass

    def __init__(self, frame):
        self.previous_frame = frame

    @abstractmethod
    def estimate_position(self, frame):
        """

        :param frame:
        """
        flow = cv2.calcOpticalFlowFarneback(self.previous_frame, frame, 0.5, 1, 5, 15, 10, 5, 1, cv2.WINDOW_AUTOSIZE)
        change_in_x = flow[:, :, 0]
        change_in_y = flow[:, :, 1]
        self.previous_frame = frame
        print(change_in_x, change_in_y)
