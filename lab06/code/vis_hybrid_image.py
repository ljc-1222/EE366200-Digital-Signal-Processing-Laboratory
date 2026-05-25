import numpy as np
import cv2

def vis_hybrid_image(hybrid_image):
    """
    Visualize a hybrid image by progressively downsampling
    and concatenating the images horizontally.
    """
    scales = 5
    scale_factor = 0.5
    padding = 5

    original_height, _, num_colors = hybrid_image.shape
    output = hybrid_image.copy()
    cur_image = hybrid_image.copy()

    for _ in range(1, scales):
        output = np.concatenate((output, np.ones((original_height, padding, num_colors))), axis=1)
        cur_image = cv2.resize(cur_image, (0, 0), fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_LINEAR)
        tmp = np.concatenate((np.ones((original_height - cur_image.shape[0], cur_image.shape[1], num_colors)), cur_image), axis=0)
        output = np.concatenate((output, tmp), axis=1)

    return output
