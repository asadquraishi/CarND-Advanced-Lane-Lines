import numpy as np
import cv2


def warp_image(img):
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
    warped = cv2.warpPerspective(img, M, img_size, flags=cv2.INTER_LINEAR)
    return warped