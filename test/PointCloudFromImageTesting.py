# coding=utf-8

from PointCloudViewer import VtkPointCloud
from Reconstruct import *


def separate_images(*args):
    image_arguments = len(args)

    assert image_arguments >= 3, "Cannot have more than 2 arguments at the moment"

    right_image = left_image = height = width = None

    if image_arguments == 1:
        image = cv2.imread(args[0])
        height, width = image.shape[:2]
        right_image, left_image = image[:, :width / 2, :], image[:, width / 2:, :]

    elif image_arguments == 2:
        left_image = cv2.imread(args[0])
        height, width = left_image.shape[:2]
        right_image = cv2.imread(args[0])

    return left_image, right_image, height, width


if __name__ == '__main__':
    with VtkPointCloud() as point_cloud:

        left, right, height, width = separate_images("aloeL.jpg", "aloeR.jpg")

        stereo = StereoBM(width, height, .8 * width, show_disparity=True)

        # while True:
        disp = stereo.compute(left, right)

        points, colors = stereo.to_3d(disp, left)
        point_cloud.clear_points()
        point_cloud.add_points(points, colors)

        while True:
            key = cv2.waitKey() & 0xFF

            if key == ord('q'):
                break

            elif key == ord(' '):
                disp = stereo.compute(left, right)

                points, colors = stereo.to_3d(disp, left)
                point_cloud.clear_points()
                point_cloud.add_points(points, colors)
        cv2.destroyAllWindows()
