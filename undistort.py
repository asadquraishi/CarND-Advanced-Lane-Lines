import cv2
import numpy as np

def warp_image(img):
    img_size = (img.shape[1], img.shape[0])
    print(img_size)
    source = np.float32([[587,451],
                        [230,701],
                        [1078,701],
                        [695,451]]) # best so far
    '''source = np.float32([[(img_size[0] / 2) - 65, (img_size[1] /2 + 100)],
                         [((img_size[0] / 6) - 10), img_size[1]],
                         [(img_size[0] * 5/6) + 60, img_size[1]],
                         [(img_size[0] / 2 + 65), img_size[1] / 2 + 100]])'''
    dest = np.float32([[320,0],
                      [320,720],
                      [960,720],
                      [960,0]])
    '''dest = np.float32([[(img_size[0] / 4), 0],
                       [(img_size[0] / 4), img_size[1]],
                       [img_size[0] * 3 /4, img_size[1]],
                       [img_size[0] * 3 / 4, 0]])'''
    M = cv2.getPerspectiveTransform(source, dest)
    warped = cv2.warpPerspective(img, M, img_size, flags=cv2.INTER_LINEAR)
    return warped