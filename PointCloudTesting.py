import numpy as np

from PointCloudViewer import start_point_cloud


def get_random_3(min, max):
    return [np.random.randint(min, max), np.random.randint(min, max), np.random.randint(min, max)]


if __name__ == '__main__':
    point_cloud, renderWindow, renderWindowInteractor = start_point_cloud()

    for i in range(1000):
        # print(i)
        values = get_random_3(-10, 10), get_random_3(0, 255)

        print(values)

        point_cloud.add_point(*values)
        renderWindow.Render()

    renderWindowInteractor.Start()
# Library to use MayAvi NThought
