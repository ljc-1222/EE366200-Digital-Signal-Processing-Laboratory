# LAB7/code/FindCorners_en.py

import numpy as np
from scipy.signal import convolve2d
from scipy.ndimage import maximum_filter
import matplotlib.pyplot as plt

def find_corners(frame, dx, dy, g, threshold, r, alpha):
    """
    Harris corner detection function.

    Input:
        frame (np.array): Input image read from source.
        dx (np.array): Horizontal gradient filter.
        dy (np.array): Vertical gradient filter.
        g (np.array): Gaussian filter.
        threshold (int): Threshold for finding local maxima (0 ~ 1000).
        r (int): Neighborhood radius used for computing RBinary.
        alpha (float): Empirical constant k for the Harris detector.

    Returns:
        tuple: (I, row, col, count)
            I (np.array): Float image converted from "frame".
            row (np.array): Row coordinates of corners.
            col (np.array): Column coordinates of corners.
            count (int): Number of detected corners.
    """
    # Convert frame to float
    I = frame.astype(float)
    
    # Handle extra batch dimension if present (shape like (1, H, W, C))
    if I.ndim == 4 and I.shape[0] == 1:
        I = np.squeeze(I, axis=0)
        frame = np.squeeze(frame, axis=0)
    
    # Show original image
    plt.figure()
    plt.imshow(frame)
    plt.title('Original Image')

    ########################### Interest Points ###########################

    # --- Compute image gradients ---

    # Grayscale conversion [cite: 33]
    if I.ndim == 3:
        # RGB image
        I_gray = I[:,:,0] * 0.299 + I[:,:,1] * 0.587 + I[:,:,2] * 0.114
    else:
        # Already grayscale
        I_gray = I

    # Compute Ix, Iy
    Ix = convolve2d(I_gray, dx, mode='same')
    Iy = convolve2d(I_gray, dy, mode='same')

    # --- Compute all components of the second-moment matrix M
    # M = [[Ix2, Ixy], [Iyx, Iy2]]
    # Note Ix2, Ixy, Iy2 are all Gaussian smoothed

    # Compute Ix2 Iy2 Ixy
    ### Your Code ###
    Ix2 = convolve2d(Ix**2, g, mode='same')
    Iy2 = convolve2d(Iy**2, g, mode='same')
    Ixy = convolve2d(Ix*Iy, g, mode='same')
    
    M = np.array([[Ix2, Ixy], [Ixy, Iy2]])
    
    # --- Visualize Ixy ---
    plt.figure()
    plt.imshow(Ixy)
    plt.title('Ixy')

    # --- Corner response R = det(M) - alpha * trace(M)^2

    # Calculate R
    ### Your code ###
    R = Ix2 * Iy2 - Ixy * Ixy - alpha * (Ix2 + Iy2)**2

    # Map the maximum of R to 1000
    R = (1000 / R.max()) * R

    # --- Use maximum_filter to implement max-filter ---
    sze = 2 * r + 1  # domain width

    # --- Find local maxima and obtain RBinary ---
    # Compute MX, RBinary 
    ### MX = maximum_filter())
    ###RBinary = 
    
    MX = maximum_filter(R, size=sze)
    RBinary = (R > threshold) & (R == MX)
    
    # --- Keep corners away from image borders ---
    offe = r
    
    # Clear R to store only final corners
    R_corners = np.zeros_like(R)
    
    # Ignore image borders and keep only middle-region corners
    R_corners[offe:-offe, offe:-offe] = RBinary[offe:-offe, offe:-offe]
    
    row, col = np.nonzero(R_corners)
    
    count = len(row)  # Count total corners

    return I, row, col, count