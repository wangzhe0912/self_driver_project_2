# -*- coding: UTF-8 -*-
import numpy as np


class Line(object):
    """
    Line is a class to define the land line
    """

    def __init__(self):
        """
        # 初始化
        """
        # 该车道线本身是继承的？还是探测的？
        self.detected = False
        # 最近几帧拟合结果
        self.recent_xfitted = []
        # 最近几帧的平均结果
        self.bestx = None
        # polynomial coefficients averaged over the last n iterations
        self.best_fit = None
        # 当前的多项式系数
        # polynomial coefficients for the most recent fit
        self.current_fit = [np.array([False])]
        # radius of curvature of the line in some units
        # 当前的曲率半径
        self.radius_of_curvature = None
        # distance in meters of vehicle center from the line
        # 当前的偏移量
        self.line_base_pos = None
        # difference in fit coefficients between last and new fits
        # 最近几次拟合系数的变化量
        self.diffs = np.array([0, 0, 0], dtype='float')
        # x values for detected line pixels
        # 检测到的所有x点
        self.allx = None
        # y values for detected line pixels
        # 检测到的所有y的点
        self.ally = None
