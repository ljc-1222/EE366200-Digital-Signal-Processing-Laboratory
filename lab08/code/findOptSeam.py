import numpy as np
import sys
import os

def findOptSeam(energy, seamDirection):
    """
    Following Avidan & Shamir (2007)
    Finds optimal seam by the given energy of an image
    Returns mask with 0 mean a pixel is in the seam    
    """

    if seamDirection == 0:  # vertical seam
        # Padding
        M = np.pad(energy, ((0, 0), (1, 1)),
                   mode='constant', constant_values=sys.float_info.max)
        h, w = M.shape
        ############### YOUR CODE HERE ###############
        # Forward accumulation
        for i in range(1, h):
            for j in range(1, w - 1):
                M[i, j] += min(M[i - 1, j - 1], M[i - 1, j], M[i - 1, j + 1])
        ############### YOUR CODE ENDS ###############
        # Find minimal energy in the last row
        idx = np.argmin(M[h - 1, :])
        seamEnergy = M[h - 1, idx]

        # Initialize for backtracking (same length as image height)
        optSeamIndexArray = np.zeros(h, dtype=np.uint32)
        optSeamIndexArray[-1] = idx

        ############### YOUR CODE HERE ###############
        # Backtrack the path of minimum seam
        for i in range(h - 2, -1, -1):
            idx = optSeamIndexArray[i + 1]
            optSeamIndexArray[i] = np.argmin(M[i, idx - 1:idx + 2]) + (idx - 1)
        ############### YOUR CODE ENDS ###############        
        
        # Remove padding offset and clamp to valid range
        optSeamIndexArray = np.clip(optSeamIndexArray - 1, 0, w - 2)
        return optSeamIndexArray, seamEnergy
    else:  # horizontal seam
        ############### YOUR CODE HERE ###############
        M = np.pad(energy, ((1, 1), (0, 0)),
                   mode='constant', constant_values=sys.float_info.max)
        h, w = M.shape
        for j in range(1, w):
            for i in range(1, h - 1):
                M[i, j] += min(M[i - 1, j - 1], M[i, j - 1], M[i + 1, j - 1])
        idx = np.argmin(M[:, w - 1])
        seamEnergy = M[idx, w - 1]
        optSeamIndexArray = np.zeros(w, dtype=np.uint32)
        optSeamIndexArray[-1] = idx
        for j in range(w - 2, -1, -1):
            idx = optSeamIndexArray[j + 1]
            optSeamIndexArray[j] = np.argmin(M[idx - 1:idx + 2, j]) + (idx - 1)
        optSeamIndexArray = np.clip(optSeamIndexArray - 1, 0, h - 2)
        ############### YOUR CODE ENDS ###############

        return optSeamIndexArray, seamEnergy
    
if __name__ == "__main__":
    import cv2
    import calcEnergy
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(os.path.dirname(base_dir), "data")
    I = cv2.imread(os.path.join(data_dir, "sea.jpg"))
    I = cv2.cvtColor(I, cv2.COLOR_BGR2RGB)
    energy = calcEnergy.calcEnergy(I)
    
    seamIndex, seamEnergy = findOptSeam(energy, 0)
    I_v_seam = I.copy()
    for i in range(len(seamIndex)):
        I_v_seam[i, seamIndex[i]] = [255, 0, 0]
    
    seamIndex, seamEnergy = findOptSeam(energy, 1)
    I_h_seam = I.copy()
    for i in range(len(seamIndex)):
        I_h_seam[seamIndex[i], i] = [255, 0, 0]
        
    cv2.imshow("Vertical Seam", cv2.cvtColor(I_v_seam, cv2.COLOR_RGB2BGR))
    cv2.imshow("Horizontal Seam", cv2.cvtColor(I_h_seam, cv2.COLOR_RGB2BGR))
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
