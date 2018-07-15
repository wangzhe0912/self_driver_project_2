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


def gradient_thresholding_filter(img, min_thresholding, max_thresholding):
    """
    #
    :param img:
    :param min_thresholding:
    :param max_thresholding:
    :return:
    """
    pass


def final_thresholding_filter(img, color_thresholding, gradient_thresholding, sobel_thresholding):
    """
    #
    :param img:
    :param color_thresholding:
    :param gradient_thresholding:
    :param sobel_thresholding:
    :return:
    """
    pass


if __name__ == "__main__":
    test_picture = "../test_images/straight_lines1.jpg"
    img = mpimg.imread(test_picture)
    # result = sobel_thresholding_filter(img, 100, 255, 3)
    # result = color_thresholding_filter(img, 200, 255)


    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
    f.tight_layout()
    ax1.imshow(img)
    ax1.set_title('Original Image', fontsize=50)
    ax2.imshow(result, cmap='gray')
    ax2.set_title('Filter Image', fontsize=50)
    plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
    plt.show()
