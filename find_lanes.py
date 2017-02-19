import pickle
import cv2
import numpy as np
import matplotlib
matplotlib.use('qt5agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

def pipeline(img, s_thresh=(100, 255), sx_thresh=(12, 255), sy_thresh=(25, 255), v_thresh=(50, 255)):
    img = np.copy(img)
    # Convert to HSV color space and separate the V channel
    hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS).astype(np.float)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float)
    l_channel = hls[:, :, 1]
    s_channel = hls[:, :, 2]
    v_channel = hsv[:, :, 2]
    # Sobel x
    sobelx = cv2.Sobel(l_channel, cv2.CV_64F, 1, 0)  # Take the derivative in x
    abs_sobelx = np.absolute(sobelx)  # Absolute x derivative to accentuate lines away from horizontal
    scaled_sobelx = np.uint8(255 * abs_sobelx / np.max(abs_sobelx))
    # Sobel y
    sobely = cv2.Sobel(l_channel, cv2.CV_64F, 1, 0)  # Take the derivative in x
    abs_sobely = np.absolute(sobely)  # Absolute x derivative to accentuate lines away from horizontal
    scaled_sobely = np.uint8(255 * abs_sobely / np.max(abs_sobely))

    # Threshold x gradient
    sxbinary = np.zeros_like(scaled_sobelx)
    sxbinary[(scaled_sobelx >= sx_thresh[0]) & (scaled_sobelx <= sx_thresh[1])] = 1

    # Threshold y gradient
    sybinary = np.zeros_like(scaled_sobely)
    sybinary[(scaled_sobely >= sy_thresh[0]) & (scaled_sobely <= sy_thresh[1])] = 1

    # Threshold S channel
    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel >= s_thresh[0]) & (s_channel <= s_thresh[1])] = 1

    # Threshold V channel
    v_binary = np.zeros_like(v_channel)
    v_binary[(v_channel >= v_thresh[0]) & (v_channel <= v_thresh[1])] = 1

    # Stack each channel
    # Note color_binary[:, :, 0] is all 0s, effectively an all black image. It might
    # be beneficial to replace this channel with something else.
    # color_binary = np.dstack(( np.zeros_like(sxbinary), sxbinary, v_binary))
    c_binary = np.zeros_like(v_channel)
    c_binary[(s_binary == 1) & (v_binary == 1)] = 1
    color_binary = np.zeros_like(img[:, :, 0])
    color_binary[((sxbinary == 1) & (sybinary == 1)) | (c_binary == 1)] = 255
    return color_binary


if __name__ == '__main__':
    # Read in the saved camera matrix and distortion coefficients
    dist_pickle = pickle.load(open("camera_cal/dist_pickle.p", "rb"))
    mtx = dist_pickle["mtx"]
    dist = dist_pickle["dist"]

    # Read in an image
    image_file = 'test_images/straight_lines1.jpg'
    img = cv2.imread(image_file)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Undistort the image
    undist = cv2.undistort(img, mtx, dist, None, mtx)

    # Colour and gradient transform
    result = pipeline(undist)

    # Plot the result
    f, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(24, 9))
    f.tight_layout()

    ax1.imshow(img)
    ax1.set_title('Original Image', fontsize=40)

    ax2.imshow(result)
    ax2.set_title('Pipeline Result', fontsize=40)

    # Warp the image
    img_size = (img.shape[1], img.shape[0])
    print(img_size)
    source = np.float32([[592, 451],
                         [230, 701],
                         [1078, 701],
                         [689, 451]])

    dest = np.float32([[320, 0],
                       [320, 720],
                       [960, 720],
                       [960, 0]])
    M = cv2.getPerspectiveTransform(source, dest)
    warped = cv2.warpPerspective(result, M, img_size, flags=cv2.INTER_LINEAR)

    ax3.imshow(warped)
    ax3.set_title('Warped', fontsize=40)
    plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
    plt.show()


    '''# Display undistorted image - uncomment this to see the lines
    f, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16, 6))
    f.tight_layout()
    ax1.imshow(img)
    ax1.set_title('Original Image', fontsize=20)
    ax2.imshow(undist)
    ax2.set_title('Undistorted Image', fontsize=20)
    sx = [source[0, 0], source[1, 0], source[2, 0], source[3, 0]]
    sy = [source[0, 1], source[1, 1], source[2, 1], source[3, 1]]
    sline = Line2D(sx, sy, color='r', marker='.')
    ax2.add_line(sline)
    ax3.imshow(warped)
    ax3.set_title('Warped Image', fontsize=20)
    dx = [dest[0, 0], dest[1, 0], dest[2, 0], dest[3, 0]]
    dy = [dest[0, 1], dest[1, 1], dest[2, 1], dest[3, 1]]
    dline = Line2D(dx, dy, color='b', marker='.')
    ax3.add_line(dline)
    plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
    plt.show()'''

