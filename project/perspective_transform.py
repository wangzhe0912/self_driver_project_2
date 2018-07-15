import cv2
import numpy as np


def get_transform_m(src, dst):
    """
    # get the perspective transform function
    :param src:
    :param dst:
    :return:
    """
    M = cv2.getPerspectiveTransform(src, dst)
    Minv = cv2.getPerspectiveTransform(dst, src)
    return M, Minv


def perspective_transform(img, m):
    """
    # execute perspective transform
    :param img:
    :param m:
    :return:
    """
    img_size = (img.shape[1], img.shape[0])
    warped = cv2.warpPerspective(img, m, img_size, flags=cv2.INTER_LINEAR)
    return warped


def reverse_picture(perspective_filter_img, left_fit, right_fit, Minv):
    """
    # 将转换图处理后的结果转换为正常视角
    :param perspective_filter_img:
    :param left_fit:
    :param right_fit:
    :param Minv:
    :return:
    """
    warp_zero = np.zeros_like(perspective_filter_img).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))
    ploty = np.linspace(0, perspective_filter_img.shape[0] - 1, perspective_filter_img.shape[0])
    left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
    right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]
    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))

    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    newwarp = cv2.warpPerspective(color_warp, Minv, (perspective_filter_img.shape[1], perspective_filter_img.shape[0]))
    return newwarp
