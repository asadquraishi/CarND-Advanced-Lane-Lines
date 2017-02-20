import pickle
import cv2
import matplotlib
import numpy as np
matplotlib.use('qt5agg')
import matplotlib.pyplot as plt
import process_image as pi
from math import trunc

# window settings
window_width = 50
window_height = 90 # Break image into 9 vertical layers since image height is 720
margin = 40 # How much to slide left and right for searching

if __name__ == '__main__':
    # Read in the saved camera matrix and distortion coefficients
    dist_pickle = pickle.load(open("camera_cal/dist_pickle.p", "rb"))
    mtx = dist_pickle["mtx"]
    dist = dist_pickle["dist"]

    # Read in an image
    image_file = 'test_images/test2.jpg'
    img = cv2.imread(image_file)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Undistort the image
    undist = cv2.undistort(img, mtx, dist, None, mtx)

    # Colour and gradient transform
    binary = pi.pipeline(undist, sx_thresh=(12, 255), sy_thresh=(25, 255), s_thresh=(100, 255), v_thresh=(50, 255))

    # Warp the binary image
    Minv, binary_warped = pi.warp_image(binary)

    #pi.plot_pipeline(img, binary, binary_warped)

    window_centroids = pi.find_window_centroids(binary_warped, window_width, window_height, margin)

    # If we found any window centers
    if len(window_centroids) > 0:

        # Points used to draw all the left and right windows
        l_points = np.zeros_like(binary_warped)
        r_points = np.zeros_like(binary_warped)

        # Go through each level and draw the windows
        for level in range(0, len(window_centroids)):
            # Window_mask is a function to draw window areas
            l_mask = pi.window_mask(window_width, window_height, binary_warped, window_centroids[level][0], level)
            r_mask = pi.window_mask(window_width, window_height, binary_warped, window_centroids[level][1], level)
            # Add graphic points from window mask here to total pixels found
            l_points[(l_points == 255) | ((l_mask == 1))] = 255
            r_points[(r_points == 255) | ((r_mask == 1))] = 255

        # Draw the results
        template = np.array(r_points + l_points, np.uint8)  # add both left and right window pixels together
        zero_channel = np.zeros_like(template)  # create a zero color channle
        template = np.array(cv2.merge((zero_channel, template, zero_channel)), np.uint8)  # make window pixels green
        warpage = np.array(cv2.merge((binary_warped, binary_warped, binary_warped)),
                           np.uint8)  # making the original road pixels 3 color channels
        output = cv2.addWeighted(warpage, 1, template, 0.5, 0.0)  # overlay the orignal road image with window results

    # If no window centers found, just display orginal road image
    else:
        output = np.array(cv2.merge((binary_warped, binary_warped, binary_warped)), np.uint8)

    '''# Display the final results
    plt.imshow(output)
    plt.title('window fitting results')'''

    # Extract left and right line pixel positions
    lefty = [window_height * (n) for n in np.arange(len(window_centroids), 0, -1)]
    leftx = [x[0] for x in window_centroids]
    rightx = [x[1] for x in window_centroids]
    righty = lefty

    # Fit a second order polynomial to each
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)

    # Generate x and y values for plotting
    ploty = np.linspace(0, binary_warped.shape[0] - 1, binary_warped.shape[0])
    left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
    right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]

    # Define conversions in x and y from pixels space to meters
    ym_per_pix = 3 / 81  # meters per pixel in y dimension
    xm_per_pix = 3.7 / 648  # meters per pixel in x dimension
    #ym_per_pix = 30 / 720  # meters per pixel in y dimension
    #xm_per_pix = 3.7 / 700  # meters per pixel in x dimension

    # Fit new polynomials to x,y in world space
    left_fit_cr = np.polyfit(np.asarray(lefty) * ym_per_pix, np.asarray(leftx) * xm_per_pix, 2)
    right_fit_cr = np.polyfit(np.asarray(righty) * ym_per_pix, np.asarray(rightx) * xm_per_pix, 2)
    # Calculate the new radii of curvature
    y_eval = np.max(ploty)
    left_curverad = ((1 + (2 * left_fit_cr[0] * y_eval * ym_per_pix + left_fit_cr[1]) ** 2) ** 1.5) / np.absolute(
        2 * left_fit_cr[0])
    right_curverad = ((1 + (2 * right_fit_cr[0] * y_eval * ym_per_pix + right_fit_cr[1]) ** 2) ** 1.5) / np.absolute(
        2 * right_fit_cr[0])
    # Now our radius of curvature is in meters
    print(left_curverad, 'm', right_curverad, 'm')

    '''plt.plot(left_fitx, ploty, color='red', linewidth=2)
    plt.plot(right_fitx, ploty, color='red', linewidth=2)
    plt.xlim(0, 1280)
    plt.ylim(720, 0)'''

    # Create an image to draw the lines on
    warp_zero = np.zeros_like(binary_warped).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))

    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    newwarp = cv2.warpPerspective(color_warp, Minv, (img.shape[1], img.shape[0]))
    # Combine the result with the original image
    result = cv2.addWeighted(undist, 1, newwarp, 0.3, 0)

    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(result, 'Left lane curvature in meters: {}'.format(trunc(left_curverad)), (50, 50), font, 0.5, (200, 255, 155),
                1, cv2.LINE_AA)
    cv2.putText(result, 'Right lane curvature in meters: {}'.format(trunc(right_curverad)), (50, 75), font, 0.5,
                (200, 255, 155), 1,
                cv2.LINE_AA)
    plt.imshow(result)
    plt.show()
