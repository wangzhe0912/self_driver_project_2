# -*- coding: UTF-8 -*-
import numpy as np
import cv2
import matplotlib.pyplot as plt


def get_polynomial_fitting_curve(binary_warped):
    """
    # use histogram to localize the land line
    :param binary_warped: input is the filter_img after perspective transform
    # 输入图像是将阈值滤波的结果图像进行投影变换后的图像
    :return:
    """
    # 计算图像出现的直方图
    histogram = np.sum(binary_warped[binary_warped.shape[0]//2:,:], axis=0)
    # 初始化一个输出图像，初始为输入图像的三次叠加且将1的点变为255
    out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255
    # 找出直方图的中位点，即图像的横中心
    midpoint = np.int(histogram.shape[0]//2)
    # 左侧车道认为是左侧直方图的最大值对应的点
    leftx_base = np.argmax(histogram[:midpoint])
    # 右侧的车道认为是右侧直方图最大值对应的点
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    # 设置整幅图像滑窗的数目
    nwindows = 9
    # 计算每个滑窗的高度
    window_height = np.int(binary_warped.shape[0]//nwindows)
    # 找出图像中所有点的坐标
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])

    # Current positions to be updated for each window
    # 设置初始窗的坐标点
    leftx_current = leftx_base
    rightx_current = rightx_base

    # Set the width of the windows +/- margin
    # 设置窗的两侧宽度为100
    margin = 100

    # Set minimum number of pixels found to recenter window
    # 设置临近窗像素点的最小值为50
    minpix = 50

    # Create empty lists to receive left and right lane pixel indices
    # 创建一个空数组用于接收左右车道的索引值
    left_lane_inds = []
    right_lane_inds = []

    # Step through the windows one by one
    # 依次遍历每个窗口
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        # 找出当前窗口对应的纵坐标的上限点和下限点
        win_y_low = binary_warped.shape[0] - (window+1)*window_height
        win_y_high = binary_warped.shape[0] - window*window_height
        # 分别找出左侧和右侧车道线的左边缘和右边缘
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin
        # Draw the windows on the visualization image
        # 在二维图像中划出该线段
        # 画出的是绿色的方框
        cv2.rectangle(out_img,(win_xleft_low,win_y_low),(win_xleft_high,win_y_high),
        (0,255,0), 2)
        cv2.rectangle(out_img,(win_xright_low,win_y_low),(win_xright_high,win_y_high),
        (0,255,0), 2)
        # Identify the nonzero pixels in x and y within the window
        # 找出在这个区域内的非零点
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
        (nonzerox >= win_xleft_low) &  (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
        (nonzerox >= win_xright_low) &  (nonzerox < win_xright_high)).nonzero()[0]
        # Append these indices to the lists
        # 将这些非零点添加至数组中
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        # If you found > minpix pixels, recenter next window on their mean position
        # 当我们找到的点的数量大于我们设置的阈值时，下一个左侧车道和右侧车道的窗的中心位置分别对应于目前这个窗中的均值位置。
        # 从而实现更新了窗的中心位置
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

    # 将左、右侧的线段分别连接起来
    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)

    # 在图像中分别显示在左右区间内出现的阈值点
    out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
    out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]

    # Extract left and right line pixel positions
    # 抽取出左侧和右侧车道线的点位置
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    # 对真实世界的尺度变换
    ym_per_pix = 30 / 720  # meters per pixel in y dimension
    xm_per_pix = 3.7 / 700  # meters per pixel in x dimension

    # 分别对左侧车道和右侧车道进行多项式拟合
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)

    ploty = np.linspace(0, binary_warped.shape[0] - 1, binary_warped.shape[0])

    # # 绘制左、右侧车道的拟合曲线
    # left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
    # right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]

    # plt.imshow(out_img)
    # # 画的是多项式拟合曲线
    # plt.plot(left_fitx, ploty, color='yellow')
    # plt.plot(right_fitx, ploty, color='yellow')
    # plt.xlim(0, 1280)
    # plt.ylim(720, 0)
    # plt.show()

    # Fit new polynomials to x,y in world space
    left_fit_cr = np.polyfit(lefty * ym_per_pix, leftx * xm_per_pix, 2)
    # Calculate the new radii of curvature
    y_eval = np.max(ploty)
    left_curverad = ((1 + (2 * left_fit_cr[0] * y_eval * ym_per_pix + left_fit_cr[1]) ** 2) ** 1.5) / np.absolute(
        2 * left_fit_cr[0])

    car_center = (left_fit[0] * 720 ** 2 + left_fit[1] * 720 + left_fit[2] +
                  right_fit[0] * 720 ** 2 + right_fit[1] * 720 + right_fit[2]) / 2
    image_center = binary_warped.shape[1] / 2
    bias_meter = (car_center - image_center) * xm_per_pix
    return left_fit, right_fit, left_curverad, bias_meter
