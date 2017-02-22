import pickle
import cv2
import matplotlib
import numpy as np
matplotlib.use('qt5agg')
import matplotlib.pyplot as plt
import process_image as pi
from moviepy.editor import VideoFileClip
from math import trunc

# window settings
window_width = 50
window_height = 90 # Break image into 9 vertical layers since image height is 720
margin = 40 # How much to slide left and right for searching
# Define conversions in x and y from pixels space to meters
ym_per_pix = 3 / 81  # meters per pixel in y dimension
xm_per_pix = 3.7 / 648  # meters per pixel in x dimension
tolerance = 0.12 # Look at differences in curvature
average_window = 5 # length of history

dist_pickle = pickle.load(open("camera_cal/dist_pickle.p", "rb"))
mtx = dist_pickle["mtx"]
dist = dist_pickle["dist"]
left_lane = pi.Line()
right_lane = pi.Line()
begun = False # First frame is treated differently
detected = False

def process_video(img):
    global begun, detected
    # Undistort the image
    undist = cv2.undistort(img, mtx, dist, None, mtx)

    # Colour and gradient transform
    binary = pi.pipeline(undist, sx_thresh=(12, 255), sy_thresh=(25, 255), s_thresh=(100, 255), v_thresh=(50, 255))

    # Warp the binary image
    Minv, binary_warped = pi.warp_image(binary)

    # prepare y points for plotting
    ploty = np.linspace(0, binary_warped.shape[0] - 1, binary_warped.shape[0])

    window_centroids = pi.find_window_centroids(binary_warped, window_width, window_height, margin)
    # Extract left and right line pixel positions
    lefty = [window_height * (n) for n in np.arange(len(window_centroids), 0, -1)]
    leftx = [x[0] for x in window_centroids]
    rightx = [x[1] for x in window_centroids]
    righty = lefty

    # Fit a second order polynomial to each
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)

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

    # Generate x and y values for plotting
    left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
    right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]

    # Compare the data with previous
    if begun:
        # Check if left lane line is recognized
        if abs(left_lane.radius_of_curvature / left_curverad - 1) > tolerance:
            left_curverad = left_lane.radius_of_curvature
            detected = False
        else:
            left_lane.radius_of_curvature = left_curverad
            detected = True

            # set left x points and average
            if len(left_lane.recent_xfitted) > average_window:
                left_lane.recent_xfitted.pop(0)
            left_lane.recent_xfitted.append(left_fitx)
            left_lane.bestx = np.mean(np.asarray(left_lane.recent_xfitted), axis=0).tolist()
            left_lane.current_fit = left_fit

        # Check if right lane line is recognized
        if abs(right_lane.radius_of_curvature / right_curverad -1) > tolerance:
            right_curverad = right_lane.radius_of_curvature
            detected = False
        else:
            right_lane.radius_of_curvature = right_curverad
            detected = True

            # set right x points and average
            if len(right_lane.recent_xfitted) > average_window:
                right_lane.recent_xfitted.pop(0)
            right_lane.recent_xfitted.append(right_fitx)
            right_lane.bestx = np.mean(np.asarray(right_lane.recent_xfitted), axis=0).tolist()
            right_lane.current_fit = right_fit

        left_fitx = left_lane.bestx
        right_fitx = right_lane.bestx
    else:
        # Treat the first frame
        left_lane.radius_of_curvature = left_curverad
        right_lane.radius_of_curvature = right_curverad
        left_lane.current_fit = left_fit
        right_lane.current_fit = right_fit
        left_lane.recent_xfitted.append(left_fitx)
        right_lane.recent_xfitted.append(right_fitx)
        left_lane.bestx = left_fitx
        right_lane.bestx = right_fitx
        detected = True
        begun = True

    # Create an image to draw the lines on
    warp_zero = np.zeros_like(binary_warped).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))

    # calculate centre and offset
    lane_centre = left_fitx[binary_warped.shape[0] - 1] + (right_fitx[binary_warped.shape[0] - 1] - left_fitx[
        binary_warped.shape[0] - 1]) / 2
    car_pos = img.shape[1] / 2
    dist_from_centre = (car_pos - lane_centre) * xm_per_pix
    position = 'left of'
    if dist_from_centre > 0:
        position = 'right of'
    elif dist_from_centre == 0:
        position = 'and on'

    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    newwarp = cv2.warpPerspective(color_warp, Minv, (img.shape[1], img.shape[0]))
    # Combine the result with the original image
    result = cv2.addWeighted(undist, 1, newwarp, 0.3, 0)
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(result, 'Left lane curvature in meters: {}'.format(trunc(left_curverad)), (50, 50), font, 0.75, (200, 255, 155), 1, cv2.LINE_AA)
    cv2.putText(result, 'Right lane curvature in meters: {}'.format(trunc(right_curverad)), (50, 80), font, 0.75, (200, 255, 155), 1,
                cv2.LINE_AA)
    cv2.putText(result, 'The car is {:.3f} meters {} centre.'.format(abs(dist_from_centre), position), (50, 110), font, 0.75,
                (200, 255, 155), 1,
                cv2.LINE_AA)
    return result

if __name__ == '__main__':
    # Read in a video
    file_out = 'project_video_with_lines.mp4'
    file_in = 'project_video.mp4'
    video_in = VideoFileClip(file_in)
    video_out = video_in.fl_image(process_video)
    video_out.write_videofile(file_out, audio=False)
