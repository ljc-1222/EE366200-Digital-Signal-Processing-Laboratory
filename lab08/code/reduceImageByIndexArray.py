import numpy as np
import os

def reduceImageByIndexArray(image, seamIndexArray, seamDirection):
    """
    Remove one seam from the image based on seam index.
    Handles vertical (0) and horizontal (1) seams safely.
    """

    h, w, ch = image.shape

    if seamDirection == 0:  # vertical seam
        imageReduced = np.zeros((h, w - 1, ch), dtype=image.dtype)
        ############### YOUR CODE HERE ###############
        # clamp seamIndexArray to valid column range
        seamIndexArray = np.clip(seamIndexArray, 0, w - 1)
        # Remove seam
        for i in range(h):
            seam_idx = seamIndexArray[i]
            # Copy pixels before the seam
            if seam_idx > 0:
                imageReduced[i, :seam_idx] = image[i, :seam_idx]
            # Copy pixels after the seam
            if seam_idx < w - 1:
                imageReduced[i, seam_idx:] = image[i, seam_idx + 1:]
        
        ############### YOUR CODE ENDS ###############
        
    else:  # horizontal seam
        h, w, ch = image.shape
        
        # # --- This section can be used or not. --- #
        # # Ensure that the length of seamIndexArray = w
        # seamIndexArray = np.asarray(seamIndexArray, dtype=np.int32)
        # if len(seamIndexArray) != w:
        #     # If the length is incorrect, resample or clip.
        #     seamIndexArray = np.round(
        #         np.linspace(seamIndexArray[0], seamIndexArray[-1], w)
        #     ).astype(np.int32)

        # # clamp values ​​are in [0, h-1]
        # seamIndexArray = np.clip(seamIndexArray, 0, h - 1)
        # # ---------------------------------------- #
        
        imageReduced = np.zeros((h - 1, w, ch), dtype=image.dtype)
        ############### YOUR CODE HERE ###############
        seamIndexArray = np.clip(seamIndexArray, 0, h - 1)
        for i in range(w):
            seam_idx = seamIndexArray[i]
            # Copy pixels before the seam
            if seam_idx > 0:
                imageReduced[:seam_idx, i, :] = image[:seam_idx, i, :]
            # Copy pixels after the seam
            if seam_idx < h - 1:
                imageReduced[seam_idx:, i, :] = image[seam_idx + 1:, i, :]
        ############### YOUR CODE ENDS ###############
        
    return imageReduced

if __name__ == "__main__":
    import cv2
    import findOptSeam
    import calcEnergy
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(os.path.dirname(base_dir), "data")
    I = cv2.imread(os.path.join(data_dir, "sea.jpg"))
    energy = calcEnergy.calcEnergy(cv2.cvtColor(I, cv2.COLOR_BGR2RGB))
    seamIndexArray, seamEnergy = findOptSeam.findOptSeam(energy, 0)
    I_reduced_v = reduceImageByIndexArray(I, seamIndexArray, 0)
    
    seamIndexArray, seamEnergy = findOptSeam.findOptSeam(energy, 1)
    I_reduced_h = reduceImageByIndexArray(I, seamIndexArray, 1)
    
    cv2.imshow("Original", I)
    cv2.imshow("Reduced Vertical", I_reduced_v)
    cv2.imshow("Reduced Horizontal", I_reduced_h)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
