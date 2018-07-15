from camera import Camera
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


if __name__ == "__main__":
    import os
    camera = Camera(9, 6)
    pictures_file_names = os.listdir("../camera_cal/")
    pictures_list = ["../camera_cal/" + filename for filename in pictures_file_names]
    camera.load_picures(pictures_list)

    test_picture = "../test_images/straight_lines1.jpg"
    img = mpimg.imread(test_picture)
    result = camera.cal_undistort(img)
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
    f.tight_layout()
    ax1.imshow(img)
    ax1.set_title('Original Image', fontsize=50)
    ax2.imshow(result)
    ax2.set_title('Undistorted Image', fontsize=50)
    plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
    plt.show()
