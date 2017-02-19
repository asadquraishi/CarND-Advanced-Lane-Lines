import pickle
import cv2
import numpy as np
import matplotlib
matplotlib.use('qt5agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import process_image as pi

if __name__ == '__main__':
    # Read in the saved camera matrix and distortion coefficients
    dist_pickle = pickle.load(open("camera_cal/dist_pickle.p", "rb"))
    mtx = dist_pickle["mtx"]
    dist = dist_pickle["dist"]

    # Read in an image
    image_file = 'test_images/test4.jpg'
    img = cv2.imread(image_file)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Undistort the image
    undist = cv2.undistort(img, mtx, dist, None, mtx)

    # Colour and gradient transform
    #result = pi.pipeline(undist, sx_thresh=(12, 255), sy_thresh=(25, 255), s_thresh=(70, 200), v_thresh=(25, 150)) # started with v=50,255 not bad
    result = pi.pipeline(undist, sx_thresh=(12, 255), sy_thresh=(25, 255), s_thresh=(70, 200), v_thresh=(25, 150))

    # Plot the result
    f, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(24, 9))
    f.tight_layout()

    ax1.imshow(img)
    ax1.set_title('Original Image', fontsize=40)

    #ax2.imshow(result, cmap="gray") # display as grayscale
    ax2.imshow(result, cmap = 'binary_r')
    ax2.set_title('Pipeline Result', fontsize=40)

    warped = pi.warp_image(result)

    ax3.imshow(warped, cmap = 'binary_r')
    ax3.set_title('Warped', fontsize=40)
    plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
    plt.show()