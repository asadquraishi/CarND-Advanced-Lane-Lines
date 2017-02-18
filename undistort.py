import pickle
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from scipy.spatial import distance

# Read in the saved camera matrix and distortion coefficients
dist_pickle = pickle.load(open("camera_cal/dist_pickle.p", "rb"))
mtx = dist_pickle["mtx"]
dist = dist_pickle["dist"]

# Read in an image
image_file = 'camera_cal/calibration4.jpg'
img = cv2.imread(image_file)
nx = 9  # the number of inside corners in x
ny = 6  # the number of inside corners in y

def corners_unwarp(img, nx, ny, mtx, dist):
    dst = cv2.undistort(img, mtx, dist, None, mtx)
    gray = cv2.cvtColor(dst, cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)

    # If corners found:
    if ret:
        image = cv2.drawChessboardCorners(dst, (nx, ny), corners, ret)
        img_size = (image.shape[1], img.shape[0])
        adj_corners = np.squeeze(corners)
        stl = adj_corners[0] # [0] - 436,114 - x,y
        s_tr = adj_corners[nx-1] # [7] - 1103,224
        sbr = adj_corners[-1] # [47] - 1075,658
        sbl = adj_corners[nx*ny-nx] # [40] - 465,768
        source = np.array([stl,s_tr,sbr,sbl])
        tl = [100,100]
        tr = [img_size[0]-100,100]
        br = [tr[0],img_size[1]-100]
        bl = [tl[0],br[1]]
        dest = np.float32([tl, tr, br, bl])
        M = cv2.getPerspectiveTransform(source, dest)
        warped = cv2.warpPerspective(image, M, img_size, flags=cv2.INTER_LINEAR)
    else:
        warped = np.zeros_like(img)
        M = 0
        print("Cannot warp the {} image".format())
    return warped, M


top_down, perspective_M = corners_unwarp(img, nx, ny, mtx, dist)
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
f.tight_layout()
ax1.imshow(img)
ax1.set_title('Original Image', fontsize=50)
ax2.imshow(top_down)
ax2.set_title('Undistorted and Warped Image', fontsize=50)
plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
plt.show()
