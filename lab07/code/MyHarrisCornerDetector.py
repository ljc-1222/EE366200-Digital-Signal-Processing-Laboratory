# My Harris detector
# The code calculates
# the Harris Feature/Interest Points (FP or IP) 
# and compute how much the rectangle is shifted between two images

# When you execute the code, the test image file will open
# then the code will print out and display the feature points.
# You can select the number of FPs by changing the variables 
## corner : significant change in all direction for a sliding window

import numpy as np
import matplotlib.pyplot as plt
import os
from skimage import io
from FindCorners import find_corners

# --- helper function to create gaussian kernel (mimics fspecial) ---
def gaussian_kernel(size, sigma):
    """Creates a 2D Gaussian kernel."""
    size = int(size) // 2
    x, y = np.mgrid[-size:size+1, -size:size+1]
    normal = 1 / (2.0 * np.pi * sigma**2)
    g =  np.exp(-((x**2 + y**2) / (2.0*sigma**2))) * normal
    return g

def rectangular_kernel(size):
    """Creates a 2D rectangular kernel."""
    return np.ones((size, size))

def hanning_kernel(size):
    """Creates a 2D Hanning kernel."""
    return np.outer(np.hanning(size), np.hanning(size))/np.sum(np.hanning(size) * np.hanning(size))

def hamming_kernel(size):
    """Creates a 2D Hamming kernel."""
    return np.outer(np.hamming(size), np.hamming(size))/np.sum(np.hamming(size) * np.hamming(size))

def blackman_kernel(size):
    """Creates a 2D Blackman kernel."""
    return np.outer(np.blackman(size), np.blackman(size))/np.sum(np.blackman(size) * np.blackman(size))

# --- Parameter settings ---
# Corner response related
sigma = 2
n_x_sigma = 6
alpha = 0.04

# Non-maximum suppression related
threshold = 20  # Should be between 0 and 1000
r = 6           # Neighborhood radius used to compute RBinary

# --- Filter kernels ---
dx = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])  # Horizontal gradient filter
dy = dx.T                                            # Vertical gradient filter

# Sobel gradient filters
sobel_dx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
sobel_dy = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])

# Scharr gradient filters
scharr_dx = np.array([[-3, 0, 3], [-10, 0, 10], [-3, 0, 3]])
scharr_dy = np.array([[-3, -10, -3], [0, 0, 0], [3, 10, 3]])

# Gradient filters list
gradient_filters = [(sobel_dx, sobel_dy), (scharr_dx, scharr_dy)]

# Gaussian filter (similar to MATLAB's fspecial)
gaussian_size = max(1, int(2 * n_x_sigma * sigma))
g = gaussian_kernel(gaussian_size, sigma)

# Rectangular filter
rectangular_size = gaussian_size
rectangular = rectangular_kernel(rectangular_size)

# Hanning filter
hanning_size = gaussian_size
hanning = hanning_kernel(hanning_size)

# filter list
filters = [rectangular, hanning]

# --- Load image ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(os.path.dirname(BASE_DIR), "data")
files = [
    os.path.join(DATA_DIR, "Im.jpg"),
    os.path.join(DATA_DIR, "Im_1.jpg"),
    os.path.join(DATA_DIR, "Im_2.jpg"),
]

for file in files:
        
    frame = io.imread(file)
    
    # Handle extra batch dimension if present
    if frame.ndim == 4 and frame.shape[0] == 1:
        frame = np.squeeze(frame, axis=0)

    # --- Call FindCorners ---
    I, r1, c1, count1 = find_corners(frame, dx, dy, g, threshold, r, alpha)

    print(f"Detected {count1} corners.")

    # --- Visualize results ---
    plt.figure()
    plt.imshow(frame)
    plt.plot(c1, r1, 'or', markersize=3)  # 'or' means red circle markers
    plt.title('Detected Corners')
    plt.axis('off')  # Turn off axes
    plt.show()

for filter in filters:
    frame = io.imread(files[0])

    # --- Call FindCorners ---
    I, r1, c1, count1 = find_corners(frame, dx, dy, filter, threshold, r, alpha)

    print(f"Detected {count1} corners.")

    # --- Visualize results ---
    plt.figure()
    plt.imshow(frame)
    plt.plot(c1, r1, 'or', markersize=3)  # 'or' means red circle markers
    plt.title('Detected Corners')
    plt.axis('off')  # Turn off axes
    plt.show()
    
for filter in gradient_filters:
    frame = io.imread(files[0])

    # --- Call FindCorners ---
    I, r1, c1, count1 = find_corners(frame, filter[0], filter[1], g, threshold, r, alpha)

    print(f"Detected {count1} corners.")

    # --- Visualize results ---
    plt.figure()
    plt.imshow(frame)
    plt.plot(c1, r1, 'or', markersize=3)  # 'or' means red circle markers
    plt.title('Detected Corners')
    plt.axis('off')  # Turn off axes
    plt.show()
