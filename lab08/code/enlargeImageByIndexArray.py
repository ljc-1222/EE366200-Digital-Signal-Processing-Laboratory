import numpy as np
import os

def enlargeImageByIndexArray(image, seamIndexArray):
    """
    Enlarge image by duplicating pixels along seam,
    using smooth interpolation (average of seam neighbors)
    to avoid visible artifacts at the seam boundary.
    """

    h, w, ch = image.shape
    imageEnlarged = np.zeros((h, w + 1, ch), dtype=image.dtype)
    ############### YOUR CODE HERE ###############
    seamIndexArray = np.clip(seamIndexArray, 0, w - 1)
    for i in range(h):
        seam_idx = seamIndexArray[i]
        
        # Copy pixels before the seam
        imageEnlarged[i, :seam_idx] = image[i, :seam_idx]
        
        # Insert interpolated pixel at seam position for smooth transition
        if seam_idx > 0 and seam_idx < w - 1:
            # Average of left and right neighbors for smooth interpolation
            imageEnlarged[i, seam_idx] = (
                (image[i, seam_idx - 1].astype(np.float32) + 
                 image[i, seam_idx + 1].astype(np.float32)) / 2
            ).astype(image.dtype)
        elif seam_idx > 0:
            # Only left neighbor available, average with seam pixel
            imageEnlarged[i, seam_idx] = (
                (image[i, seam_idx - 1].astype(np.float32) + 
                 image[i, seam_idx].astype(np.float32)) / 2
            ).astype(image.dtype)
        elif seam_idx < w - 1:
            # Only right neighbor available, average with seam pixel
            imageEnlarged[i, seam_idx] = (
                (image[i, seam_idx].astype(np.float32) + 
                 image[i, seam_idx + 1].astype(np.float32)) / 2
            ).astype(image.dtype)
        else:
            # Edge case: use seam pixel itself
            imageEnlarged[i, seam_idx] = image[i, seam_idx]
        
        # Copy the original seam pixel
        imageEnlarged[i, seam_idx + 1] = image[i, seam_idx]
        
        # Copy pixels after the seam
        if seam_idx < w - 1:
            imageEnlarged[i, seam_idx + 2:] = image[i, seam_idx + 1:]
    
    ############### YOUR CODE ENDS ###############

    return imageEnlarged

# def enlargeImageByIndexArray(image, seamIndexArray):
#     """
#     Enlarge image by duplicating pixels along seam,
#     using smooth interpolation (average of seam neighbors)
#     to avoid visible artifacts at the seam boundary.
#     """

#     h, w, ch = image.shape
#     imageEnlarged = np.zeros((h, w + 1, ch), dtype=image.dtype)
#     ############### YOUR CODE HERE ###############
#     seamIndexArray = np.clip(seamIndexArray, 0, w - 1)
#     for i in range(h):
#         seam_idx = seamIndexArray[i]
        
#         # Copy pixels before the seam
#         imageEnlarged[i, :seam_idx] = image[i, :seam_idx]
        
#         # Insert exactly the same pixel as the seam pixel
#         imageEnlarged[i, seam_idx] = image[i, seam_idx]
        
#         # Copy pixels after the seam
#         if seam_idx < w - 1:
#             imageEnlarged[i, seam_idx + 1:] = image[i, seam_idx:]
    
#     ############### YOUR CODE ENDS ###############

#     return imageEnlarged

if __name__ == "__main__":
    import cv2
    import findOptSeam
    import calcEnergy
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(os.path.dirname(base_dir), "data")
    I = cv2.imread(os.path.join(data_dir, "sea.jpg"))
    energy = calcEnergy.calcEnergy(cv2.cvtColor(I, cv2.COLOR_BGR2RGB))
    seamIndexArray, seamEnergy = findOptSeam.findOptSeam(energy, 0)
    I_enlarged = enlargeImageByIndexArray(I, seamIndexArray)
    cv2.imshow("Original", I)
    cv2.imshow("Enlarged", I_enlarged)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
