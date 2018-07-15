# -*- coding: UTF-8 -*-
from camera import Camera
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from thresholding_filter import final_thresholding_filter
from perspective_transform import get_transform_m, perspective_transform, reverse_picture
from histogram import get_polynomial_fitting_curve


if __name__ == "__main__":
    import os
    camera = Camera(9, 6)
    pictures_file_names = os.listdir("../camera_cal/")
    pictures_list = ["../camera_cal/" + filename for filename in pictures_file_names]
    camera.load_picures(pictures_list)

    # test_picture = "../test_images/test1.jpg"
    test_picture = "../test_images/test2.jpg"

    img = mpimg.imread(test_picture)
    undistort_img = camera.cal_undistort(img)
    filter_result = final_thresholding_filter(undistort_img, [200, 255], [], [100, 255], sobel_kernal_size=3)

    src = np.float32([[251.9, 688], [585.655, 455.895], [692.7, 455.895], [1054, 688]])
    dst = np.float32([[300, 700], [300, 100], [900, 100], [900, 700]])

    img_size = (img.shape[1], img.shape[0])
    M, Minv = get_transform_m(src, dst)
    perspective_filter_img = perspective_transform(filter_result, M)
    # 得到的是将阈值图像透视变换后的二值图像

    left_fit, right_fit, left_curverad, bias_meter = get_polynomial_fitting_curve(perspective_filter_img)
    # left_curverad 曲率
    # bias_meter中心偏移量

    newwarp = reverse_picture(perspective_filter_img, left_fit, right_fit, Minv)
    # Combine the result with the original image
    result = cv2.addWeighted(undistort_img, 1, newwarp, 0.3, 0)
    plt.imshow(result)
    message_left_curverad = "Radius of Curvature = " + str(round(left_curverad, 1)) + "(m)"
    if bias_meter < 0:
        message_bias_meter = "Vehicle is " + str(np.abs(round(bias_meter, 3))) + "m right of center"
    else:
        message_bias_meter = "Vehicle is " + str(np.abs(round(bias_meter, 3))) + "m left of center"
    plt.text(10, 50, message_left_curverad, fontsize=8)
    plt.text(10, 100, message_bias_meter, fontsize=8)
    plt.show()
