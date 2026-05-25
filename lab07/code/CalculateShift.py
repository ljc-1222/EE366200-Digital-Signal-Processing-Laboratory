# Compute how much the rectangle is shifted between two images

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

# --- Parameter settings ---
# Corner response related
sigma = 2
n_x_sigma = 6
alpha = 0.04

# Non-maximum suppression related
threshold = 20  # Should be between 0 and 1000
r = 6           # Neighborhood radius used when computing RBinary

# --- Filter kernels ---
dx = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])  # Horizontal gradient filter
dy = dx.T                                            # Vertical gradient filter

# Gaussian filter
gaussian_size = max(1, int(2 * n_x_sigma * sigma))
g = gaussian_kernel(gaussian_size, sigma)

# --- Load images ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(os.path.dirname(BASE_DIR), "data")
frame1 = io.imread(os.path.join(DATA_DIR, 'img_1.png'))
frame2 = io.imread(os.path.join(DATA_DIR, 'img_2.png'))


# --- Call FindCorners separately for each image ---
### Your Code ###
I1, row1, col1, count1 = find_corners(frame1, dx, dy, g, threshold, r, alpha)
I2, row2, col2, count2 = find_corners(frame2, dx, dy, g, threshold, r, alpha)

# Display the detected corners in frame1
# Hint: To show these two images, please refer to the MyHarrisCornerDetector.m
### Your Code ###
plt.figure()
plt.imshow(frame1)
plt.plot(col1, row1, 'or', markersize=3)  # 'or' means red circle markers
plt.title('Detected Corners')
plt.axis('off')  # Turn off axes

# --- Visualize detected corners in frame2 ---
### Your Code ###
plt.figure()
plt.imshow(frame2)
plt.plot(col2, row2, 'or', markersize=3)  # 'or' means red circle markers
plt.title('Detected Corners')
plt.axis('off')  # Turn off axes

# --- Compute shift ---
### Your Code ###
shift_row = np.abs(row2[0] - row1[0])
shift_col = np.abs(col2[0] - col1[0])
   
# Show the rectangle's displacement between the two images
print(f"\nRow shifted: {shift_row:.2f}")
print(f"Col shifted: {shift_col:.2f}")

plt.tight_layout()
plt.show()
