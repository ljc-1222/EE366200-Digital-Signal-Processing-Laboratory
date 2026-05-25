import numpy as np

def reduceMaskByIndexArray(mask, seamIndexArray, seamDirection):
    """
    Reduce mask (logical matrix) along seam
    """
    h, w = mask.shape

    if seamDirection == 0:
        maskReduced = np.zeros((h, w - 1), dtype=bool)
        ############### YOUR CODE HERE ###############
        seamIndexArray = np.clip(seamIndexArray, 0, w - 1)
        for i in range(h):
            seam_idx = seamIndexArray[i]
            if seam_idx > 0:
                maskReduced[i, :seam_idx] = mask[i, :seam_idx]
            if seam_idx < w - 1:
                maskReduced[i, seam_idx:] = mask[i, seam_idx + 1:]
        ############### YOUR CODE ENDS ###############
        
    else:
        maskReduced = np.zeros((h - 1, w), dtype=bool)
        ############### YOUR CODE HERE ###############
        seamIndexArray = np.clip(seamIndexArray, 0, h - 1)
        for i in range(w):
            seam_idx = seamIndexArray[i]
            if seam_idx > 0:
                maskReduced[:seam_idx, i] = mask[:seam_idx, i]
            if seam_idx < h - 1:
                maskReduced[seam_idx:, i] = mask[seam_idx + 1:, i]
        ############### YOUR CODE ENDS ###############
        
    return maskReduced
