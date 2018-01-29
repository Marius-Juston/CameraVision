# coding=utf-8
import numpy as np

from PointCloudViewer import VtkPointCloud


def get_random_3(min_value, max_value):
    """

    :param min_value: 
    :param max_value: 
    :return: 
    """
    return [np.random.randint(min_value, max_value), np.random.randint(min_value, max_value),
            np.random.randint(min_value, max_value)]


if __name__ == '__main__':
    with VtkPointCloud() as point_cloud:
        for i in range(1000):
            # print(i)
            values = get_random_3(-10, 10), get_random_3(0, 255)

            print(values)

            point_cloud.add_point(*values)

# Library to use MayAvi NThought
