import pickle
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('qt5agg')
import undistort


if __name__ == '__main__':
    # Read in the saved camera matrix and distortion coefficients
    dist_pickle = pickle.load(open("camera_cal/dist_pickle.p", "rb"))
    mtx = dist_pickle["mtx"]
    dist = dist_pickle["dist"]

    # Read in an image
    image_file = 'test_images/test6.jpg'
    img = cv2.imread(image_file)

    # Undistort the image
    undist = cv2.undistort(img, mtx, dist, None, mtx)

    # Display undistorted image
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    f.tight_layout()
    ax1.imshow(img)
    ax1.set_title('Original Image', fontsize=50)
    ax2.imshow(top_down)
    ax2.set_title('Undistorted Image', fontsize=50)
    plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
    plt.show()