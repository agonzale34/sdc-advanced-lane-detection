import matplotlib.pyplot as plt
import numpy as np
import cv2


# helper to show 2 images next to each other
def visualize_result(original, result, gray=False, show=True):
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
    ax1.imshow(original)
    ax1.set_title('Original Image', fontsize=30)
    if gray:
        ax2.imshow(result, cmap='gray')
    else:
        ax2.imshow(result)
    ax2.set_title('Result Image', fontsize=30)
    if show:
        plt.show()


def visualize_result4(img1, img2, img3, img4):
    f, (ax1, ax2) = plt.subplots(2, 2, figsize=(20, 10))
    ax1[0].imshow(img1[0])
    ax1[0].set_title(img1[1], fontsize=30)
    ax2[0].imshow(img2[0])
    ax2[0].set_title(img2[1], fontsize=30)
    ax1[1].imshow(img3[0])
    ax1[1].set_title(img3[1], fontsize=30)
    ax2[1].imshow(img4[0])
    ax2[1].set_title(img4[1], fontsize=30)
    plt.show()


def show_lines_on_image_test(img_binary, left_lane, right_lane, sa_margin):
    # Create an image to draw on and an image to show the selection window
    out_img = np.dstack((img_binary, img_binary, img_binary)) * 255
    window_img = np.zeros_like(out_img)

    # Generate a polygon to illustrate the search window area
    # And recast the x and y points into usable format for cv2.fillPoly()
    left_line_window1 = np.array([np.transpose(np.vstack([left_lane.allx - sa_margin, left_lane.ally]))])
    left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_lane.allx + sa_margin,
                                                                    left_lane.ally])))])
    left_line_pts = np.hstack((left_line_window1, left_line_window2))
    right_line_window1 = np.array([np.transpose(np.vstack([right_lane.allx - sa_margin, right_lane.ally]))])
    right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_lane.allx + sa_margin,
                                                                     right_lane.ally])))])
    right_line_pts = np.hstack((right_line_window1, right_line_window2))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(window_img, np.int_([left_line_pts]), (0, 255, 0))
    cv2.fillPoly(window_img, np.int_([right_line_pts]), (0, 255, 0))
    result = cv2.addWeighted(out_img, 1, window_img, 0.3, 0)

    # Plot the polynomial lines onto the image
    plt.plot(left_lane.allx, left_lane.ally, color='yellow')
    plt.plot(right_lane.allx, right_lane.ally, color='yellow')
    plt.imshow(result)
