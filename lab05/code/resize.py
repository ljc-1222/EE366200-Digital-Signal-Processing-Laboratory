import numpy as np
from common import load_image, data_path
import matplotlib.pyplot as plt

def resize_image(img, scale, background_color=(0, 0, 0)):
    """
    Resize (nearest-neighbor, backward warping; lecture-style 1-based geometry).
    Parameters
    ----------
    img             : uint8 array (H, W, 3)
    scale           : float > 0, e.g., 1.5 or 0.5
    background_color: tuple (R, G, B) for areas not covered by original image

    Returns
    -------
    out : uint8 array (H', W', 3)
    """

    # get height, width, channel of image
    h, w, _ = img.shape

    ## step 1. calculate new width and height (use ceil to cover the whole image)
    h_new = np.ceil(h * scale).astype(int) 
    w_new = np.ceil(w * scale).astype(int)

    ## step 2. initialize output arrays with background color
    out = np.full((h_new, w_new, 3), background_color, dtype=np.uint8)

    ## backward warping using nearest-neighbor interpolation
    # for each pixel on the resized image, find the corresponding pixel in source
    for y_new in range(1, h_new + 1):
        for x_new in range(1, w_new + 1):
            # step 3. scale the new pixel (y_new, x_new) back to (y_old, x_old)
            ### Your code ###
            y_old = y_new / scale
            x_old = x_new / scale

            # step 4. Find the nearest pixel (use round)
            y_nearest = round(y_old)
            x_nearest = round(x_old)
            
            # Assign pixels from (y_nearest, x_nearest) to (y_new, x_new)
            if (1 <= y_nearest <= h) and (1 <= x_nearest <= w):
                ### Your code ###
                out[y_new - 1, x_new - 1] = img[y_nearest - 1, x_nearest - 1]


    return out

if __name__ == "__main__":
    img = load_image(data_path("image.jpg"))
    
    print(f"Image shape: {img.shape}")
    
    # Test of scale greater than 1
    img_resized = resize_image(img, scale=1.5)
    print(f"Image shape of scale 1.5: {img_resized.shape}")
    plt.figure() 
    plt.title("Scale 1.5")
    plt.imshow(img_resized)
    plt.show(block=False) 
    
    # Test of scale less than 1
    img_resized = resize_image(img, scale=0.5)
    print(f"Image shape of scale 0.5: {img_resized.shape}")
    plt.figure()  
    plt.title("Scale 0.5")
    plt.imshow(img_resized)
    plt.show(block=False) 
    
    # Test of scale equal to 1
    img_resized = resize_image(img, scale=1.0)
    print(f"Image shape of scale 1.0: {img_resized.shape}")
    plt.figure()  
    plt.title("Scale 1.0")
    plt.imshow(img_resized)
    plt.show() 
