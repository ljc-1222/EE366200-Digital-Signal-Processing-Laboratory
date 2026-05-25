import numpy as np

def my_imfilter(image, filter):
    """
    This function performs image filtering
    without using any built-in convolution functions.
    
    Input:
        image  - an H x W x C numpy array (float32 or float64)
        filter - a 2D numpy array (odd-sized kernel)
        
    Output:
        output - a filtered image with the same size as input
    """
    ############### YOUR CODE HERE ###############
    # Processing color images using a three-dimensional array
    if image.max() > 255.0:
        image = np.clip(image, 0.0, 255.0)
    elif image.min() > 1.0:
        image = np.clip(image, 0.0, 1.0)
        
    output = np.zeros_like(image)
    
    # Use np.pad() to add padding to the image then show the shape of the padded image
    # Calculate padding amounts
    pad_h = filter.shape[0] // 2
    pad_w = filter.shape[1] // 2
    
    # Pad the entire image at once, maintaining the (H, W, C) shape
    padded_image = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w), (0, 0)), mode='constant')
    
    # Use nested loops to apply the filter to the image
    for h in range(image.shape[0]):
        for w in range(image.shape[1]):
            for c in range(image.shape[2]):
                output[h, w, c] = np.sum(padded_image[h:h+filter.shape[0], w:w+filter.shape[1], c] * filter)
    
    ############### YOUR CODE END ################
    return output
