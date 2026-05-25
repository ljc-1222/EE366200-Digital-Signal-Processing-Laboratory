import numpy as np
import cv2
from my_imfilter import my_imfilter
from vis_hybrid_image import vis_hybrid_image
import matplotlib.pyplot as plt
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LAB_DIR = os.path.dirname(BASE_DIR)
DATA_DIR = os.path.join(LAB_DIR, "data")
RESULTS_DIR = os.path.join(LAB_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

file_low = [
    os.path.join(DATA_DIR, "dog.jpg"),
    os.path.join(DATA_DIR, "marilyn.jpg"),
    os.path.join(DATA_DIR, "bird.jpg"),
    os.path.join(DATA_DIR, "submarine.jpg"),
]
file_high = [
    os.path.join(DATA_DIR, "cat.jpg"),
    os.path.join(DATA_DIR, "einstein.jpg"),
    os.path.join(DATA_DIR, "plane.jpg"),
    os.path.join(DATA_DIR, "fish.jpg"),
]


for i in range(len(file_low)):
    ## Setup
    image1 = cv2.imread(file_low[i])
    image2 = cv2.imread(file_high[i])

    image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB) / 255.0
    image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2RGB) / 255.0

    # faster
    image1 = cv2.resize(image1, (0, 0), fx=0.5, fy=0.5)
    image2 = cv2.resize(image2, (0, 0), fx=0.5, fy=0.5)

    ## Filtering and Hybrid Image Constructio
    laplacian_filter = np.array([[0, 1, 0],
                                 [1, -4, 1],
                                 [0, 1, 0]], dtype=np.float32)
    
    ############### YOUR CODE HERE ###############
    # Laplacian filter
    low_frequencies_laplacian_reversed  = image1 - my_imfilter(image1, laplacian_filter)
    high_frequencies_laplacian_reversed = my_imfilter(image2, laplacian_filter)
    hybrid_image_laplacian_reversed     = 1 * low_frequencies_laplacian_reversed + 5 * high_frequencies_laplacian_reversed
    
    ############### YOUR CODE END ################
    ## Visualize and Save
    plt.figure(1); plt.imshow(np.clip(low_frequencies_laplacian_reversed, 0.0, 1.0)); plt.title('Low Frequencies Laplacian Reversed')
    plt.figure(2); plt.imshow(np.clip(high_frequencies_laplacian_reversed + 0.5, 0.0, 1.0)); plt.title('High Frequencies Laplacian Reversed (+0.5)')
    vis = vis_hybrid_image(np.clip(hybrid_image_laplacian_reversed, 0.0, 1.0))
    plt.figure(3); plt.imshow(vis); plt.title('Hybrid Image Scales Laplacian Reversed')
    plt.figure(4); plt.imshow(np.clip(hybrid_image_laplacian_reversed, 0.0, 1.0)); plt.title('Hybrid Image Laplacian Reversed')
   
    cv2.imwrite(os.path.join(RESULTS_DIR, f'low_frequencies_laplacian_reversed{i}.jpg'), (np.clip(low_frequencies_laplacian_reversed, 0.0, 1.0)[:, :, ::-1] * 255).astype(np.uint8))
    cv2.imwrite(os.path.join(RESULTS_DIR, f'high_frequencies_laplacian_reversed{i}.jpg'), (np.clip(high_frequencies_laplacian_reversed + 0.5, 0.0, 1.0)[:, :, ::-1] * 255).astype(np.uint8))
    cv2.imwrite(os.path.join(RESULTS_DIR, f'hybrid_image_laplacian_reversed{i}.jpg'), (np.clip(hybrid_image_laplacian_reversed, 0.0, 1.0)[:, :, ::-1] * 255).astype(np.uint8))
    cv2.imwrite(os.path.join(RESULTS_DIR, f'hybrid_image_scales_laplacian_reversed{i}.jpg'), (vis[:, :, ::-1] * 255).astype(np.uint8))
    plt.show()
