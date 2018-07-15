import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


class Camera(object):
    """
    Camera is a class to define the camera
    """
    def __init__(self, nx, ny):
        """
        # initialize
        :param nx:
        :param ny:
        """
        self.nx = nx
        self.ny = ny
        self.objpoints = None
        self.imgpoints = None

    def _generate_array(self, length):
        """
        # generate imgpoints array
        # import by offical tutorial
        :param length:
        :return:
        """
        result = []
        objp = np.zeros((self.nx * self.ny, 3), np.float32)
        objp[:, :2] = np.mgrid[0:self.nx, 0:self.ny].T.reshape(-1, 2)
        for i in range(length):
            result.append(objp)
        return result

    def load_picures(self, images):
        """
        # load picture to adjust the mtx and dist
        :param images:
        :return:
        """
        imgpoints = []
        for filename in images:
            # get an image
            img = mpimg.imread(filename)
            # transform to gray image
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            # find the image color
            ret, corners = cv2.findChessboardCorners(gray, (self.nx, self.ny), None)
            if ret:
                # calculate the mtx and dist
                imgpoints.append(corners)
        self.objpoints = self._generate_array(len(imgpoints))
        self.imgpoints = imgpoints

    def cal_undistort(self, img):
        """
        # input distort image
        # and calculate undistort image by mtx and dist
        :param img:  the output of cv2.imread
        :return:
        """
        if self.objpoints is None or self.imgpoints is None:
            raise Exception("this camera is not to adjust, so cannot use to cal_undistort image")
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
            self.objpoints, self.imgpoints, gray.shape[::-1], None, None
        )
        undist_img = cv2.undistort(img, mtx, dist, None, mtx)
        return undist_img


if __name__ == "__main__":
    import os
    camera = Camera(9, 6)
    pictures_file_names = os.listdir("../camera_cal/")
    pictures_list = ["../camera_cal/" + filename for filename in pictures_file_names]
    # pictures_list = ["../camera_cal/calibration2.jpg"]
    camera.load_picures(pictures_list)
    test_picture = "../test_images/test1.jpg"
    img = mpimg.imread(test_picture)
    result = camera.cal_undistort(img)
    plt.imshow(result)
    plt.axis('off')
    plt.savefig("test1_camera_fix.jpg")
    # f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
    # f.tight_layout()
    # ax1.imshow(img)
    # ax1.set_title('Original Image', fontsize=50)
    # ax2.imshow(result)
    # ax2.set_title('Undistorted Image', fontsize=50)
    # plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
    # plt.show()
