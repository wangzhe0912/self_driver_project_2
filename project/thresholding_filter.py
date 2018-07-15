import numpy as np
import cv2
import matplotlib.image as mpimg
import matplotlib.pyplot as plt


def color_thresholding_filter(img, min_thresholding, max_thresholding):
    """
    #
    :param img:
    :param min_thresholding:
    :param max_thresholding:
    :return:
    """
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    s = hls[:, :, 2]
    result = np.zeros_like(s)
    result[(s >= min_thresholding) & (s <= max_thresholding)] = 1
    return result


def sobel_thresholding_filter(img, min_thresholding, max_thresholding, sobel_kernel):
    """
    #
    :param img:
    :param min_thresholding:
    :param max_thresholding:
    :return:
    """
    # convert color image to gray
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    abs_sobel = np.power(np.power(sobelx, 2) + np.power(sobely, 2), 0.5)
    scaled_sobel = np.uint8(255 * abs_sobel / np.max(abs_sobel))
    # initial zeros array
    result = np.zeros_like(scaled_sobel)
    result[(scaled_sobel >= min_thresholding) & (scaled_sobel <= max_thresholding)] = 1
    return result


def gradient_thresholding_filter(img, min_thresholding, max_thresholding, sobel_kernel):
    """
    #
    :param img:
    :param min_thresholding:
    :param max_thresholding:
    :param sobel_kernel:
    :return:
    """
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    abs_sobelx = np.abs(sobelx)
    abs_sobely = np.abs(sobely)
    gradient_sobel = np.arctan2(abs_sobely, abs_sobelx)
    result = np.zeros_like(gradient_sobel)
    result[(gradient_sobel >= min_thresholding) & (gradient_sobel <= max_thresholding)] = 1
    return result


def final_thresholding_filter(img, color_thresholding, gradient_thresholding, sobel_thresholding, sobel_kernal_size=3):
    """
    #
    :param img:
    :param color_thresholding:
    :param gradient_thresholding:
    :param sobel_thresholding:
    :param sobel_kernal_size:
    :return:
    """
    condition = (0 == 1)
    if color_thresholding:
        color_result = color_thresholding_filter(img, color_thresholding[0], color_thresholding[1])
        combined_result = np.zeros_like(color_result)
        condition = condition | (color_result == 1)
    if gradient_thresholding:
        gradient_result = gradient_thresholding_filter(img, gradient_thresholding[0], gradient_thresholding[1], sobel_kernal_size)
        combined_result = np.zeros_like(gradient_result)
        condition = condition | (gradient_result == 1)
    if sobel_thresholding:
        sobel_result = sobel_thresholding_filter(img, sobel_thresholding[0], sobel_thresholding[1], sobel_kernal_size)
        combined_result = np.zeros_like(sobel_result)
        condition = condition | (sobel_result == 1)

    try:
        combined_result[condition] = 1
    except Exception as e:
        raise Exception(str(e))
    return combined_result


if __name__ == "__main__":
    test_picture = "../test_images/straight_lines1.jpg"
    img = mpimg.imread(test_picture)
    # result = color_thresholding_filter(img, 200, 255)
    # result = sobel_thresholding_filter(img, 100, 255, 3)
    # result = gradient_thresholding_filter(img, 0.5, 1, 15)
    result = final_thresholding_filter(img, [200, 255], [], [100, 255], sobel_kernal_size=3)
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
    f.tight_layout()
    ax1.imshow(img)
    ax1.set_title('Original Image', fontsize=50)
    ax2.imshow(result, cmap='gray')
    ax2.set_title('Filter Image', fontsize=50)
    plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
    plt.show()
