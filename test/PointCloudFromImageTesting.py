# coding=utf-8

from PointCloudViewer import VtkPointCloud
from Reconstruct import *


def separate_images(*args):
    image_arguments = len(args)
    right_image = left_image = height = width = None

    if image_arguments == 1:
        image = cv2.imread(args[0])
        height, width = image.shape[:2]
        right_image, left_image = image[:, :width / 2, :], image[:, width / 2:, :]

    elif image_arguments == 2:
        left_image = cv2.imread(args[0])
        height, width = left_image.shape[:2]
        right_image = cv2.imread(args[1])
    elif image_arguments >= 3:
        raise ValueError("Cannot have more than 2 values for the instance")
    return left_image, right_image, height, width


if __name__ == '__main__':
    with VtkPointCloud() as point_cloud:

        left, right, height, width = separate_images("aloeL.jpg", "aloeR.jpg")
        # left, right, height, width = separate_images("place1.jpg", "place2.jpg")
        # left, right, height, width = separate_images("stereo image.jpg")
        print(height, width)

        stereo = StereoBM(width, height, .8 * width, show_disparity=True, num_disparities=112, block_size=23)
        # stereo = StereoSGBM(width, height, .8 * width, show_disparity=True)
        # while True:
        disp = stereo.compute(left, right)

        cv2.imshow("Show", left)

        points, colors = stereo.to_3d(disp, left)
        point_cloud.clear_points()
        point_cloud.add_points(points, colors)

        while True:

            # elif key == ord(' '):
            disp = stereo.compute(left, right)

            points, colors = stereo.to_3d(disp, left)
            point_cloud.clear_points()
            point_cloud.add_points(points, colors)

            key = cv2.waitKey() & 0xFF

            if key == ord('q'):
                break

        cv2.destroyAllWindows()
