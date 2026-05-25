import numpy as np
import cv2
from my_imfilter import my_imfilter
from vis_hybrid_image import vis_hybrid_image
import matplotlib.pyplot as plt
import os
import time

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
    os.path.join(RESULTS_DIR, "aligned_2.jpg"),
    os.path.join(RESULTS_DIR, "aligned_4.jpg"),
]
file_high = [
    os.path.join(DATA_DIR, "cat.jpg"),
    os.path.join(DATA_DIR, "einstein.jpg"),
    os.path.join(DATA_DIR, "plane.jpg"),
    os.path.join(DATA_DIR, "fish.jpg"),
    os.path.join(RESULTS_DIR, "aligned_1.jpg"),
    os.path.join(RESULTS_DIR, "aligned_3.jpg"),
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

    ## Filtering and Hybrid Image Construction
    cutoff_frequency = [3, 2, 4, 2, 2, 4.5]

    ksize = int(cutoff_frequency[i] * 4 + 1)
    filter = cv2.getGaussianKernel(ksize, cutoff_frequency[i])
    filter = filter @ filter.T

    # Ensure the box filter has the same size as the Gaussian filter
    box_filter = np.ones((ksize, ksize), dtype=np.float32) / (ksize * ksize)
    ############### YOUR CODE HERE ###############
    # Gaussian filter
    low_frequencies_gaussian  = my_imfilter(image1, filter)
    high_frequencies_gaussian = image2 - my_imfilter(image2, filter)
    hybrid_image_gaussian     = low_frequencies_gaussian + high_frequencies_gaussian
    
    # Box filter
    low_frequencies_box  = my_imfilter(image1, box_filter)
    high_frequencies_box = image2 - my_imfilter(image2, box_filter)
    hybrid_image_box     = low_frequencies_box + high_frequencies_box
    ############### YOUR CODE END ################

    ## Visualize and Save
    plt.figure(1); plt.imshow(np.clip(low_frequencies_gaussian, 0.0, 1.0)); plt.title('Low Frequencies Gaussian')
    plt.figure(2); plt.imshow(np.clip(high_frequencies_gaussian + 0.5, 0.0, 1.0)); plt.title('High Frequencies Gaussian (+0.5)')
    plt.figure(3); plt.imshow(np.clip(low_frequencies_box, 0.0, 1.0)); plt.title('Low Frequencies Box')
    plt.figure(4); plt.imshow(np.clip(high_frequencies_box + 0.5, 0.0, 1.0)); plt.title('High Frequencies Box (+0.5)')
    vis = vis_hybrid_image(hybrid_image_gaussian)
    vis_box = vis_hybrid_image(hybrid_image_box)
    plt.figure(5); plt.imshow(np.clip(vis, 0.0, 1.0)); plt.title('Hybrid Image Scales Gaussian')
    plt.figure(6); plt.imshow(np.clip(vis_box, 0.0, 1.0)); plt.title('Hybrid Image Scales Box')

    cv2.imwrite(os.path.join(RESULTS_DIR, f'low_frequencies_gaussian{i}.jpg'), (np.clip(low_frequencies_gaussian, 0.0, 1.0)[:, :, ::-1] * 255).astype(np.uint8))
    cv2.imwrite(os.path.join(RESULTS_DIR, f'high_frequencies_gaussian{i}.jpg'), (np.clip(high_frequencies_gaussian + 0.5, 0.0, 1.0)[:, :, ::-1] * 255).astype(np.uint8))
    cv2.imwrite(os.path.join(RESULTS_DIR, f'hybrid_image_gaussian{i}.jpg'), (np.clip(hybrid_image_gaussian, 0.0, 1.0)[:, :, ::-1] * 255).astype(np.uint8))
    cv2.imwrite(os.path.join(RESULTS_DIR, f'hybrid_image_scales_gaussian{i}.jpg'), (np.clip(vis, 0.0, 1.0)[:, :, ::-1] * 255).astype(np.uint8))
    cv2.imwrite(os.path.join(RESULTS_DIR, f'low_frequencies_box{i}.jpg'), (np.clip(low_frequencies_box, 0.0, 1.0)[:, :, ::-1] * 255).astype(np.uint8))
    cv2.imwrite(os.path.join(RESULTS_DIR, f'high_frequencies_box{i}.jpg'), (np.clip(high_frequencies_box + 0.5, 0.0, 1.0)[:, :, ::-1] * 255).astype(np.uint8))
    cv2.imwrite(os.path.join(RESULTS_DIR, f'hybrid_image_box{i}.jpg'), (np.clip(hybrid_image_box, 0.0, 1.0)[:, :, ::-1] * 255).astype(np.uint8))
    cv2.imwrite(os.path.join(RESULTS_DIR, f'hybrid_image_scales_box{i}.jpg'), (np.clip(vis_box, 0.0, 1.0)[:, :, ::-1] * 255).astype(np.uint8))
    plt.show()
    
# # Store images for different sigma values
# sigma_values = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
# low_freq_images = []
# high_freq_images = []
# hybrid_images = []

# # First, generate all images
# print("Generating images for different sigma values...")
# for i in range(15):
#     ## Setup
#     image1 = cv2.imread(file_low[3])
#     image2 = cv2.imread(file_high[3])

#     image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB) / 255.0
#     image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2RGB) / 255.0

#     # faster
#     image1 = cv2.resize(image1, (0, 0), fx=0.5, fy=0.5)
#     image2 = cv2.resize(image2, (0, 0), fx=0.5, fy=0.5)

#     ## Filtering and Hybrid Image Construction
#     cutoff_frequency = sigma_values[i]  

#     ksize = int(cutoff_frequency * 4 + 1)
#     filter = cv2.getGaussianKernel(ksize, cutoff_frequency)
#     filter = filter @ filter.T
    
#     low_frequencies_gaussian  = my_imfilter(image1, filter)
#     high_frequencies_gaussian = image2 - my_imfilter(image2, filter)
#     hybrid_image_gaussian     = low_frequencies_gaussian + high_frequencies_gaussian

#     # Store images
#     low_freq_images.append(np.clip(low_frequencies_gaussian, 0.0, 1.0))
#     high_freq_images.append(np.clip(high_frequencies_gaussian + 0.5, 0.0, 1.0))
#     hybrid_images.append(np.clip(hybrid_image_gaussian, 0.0, 1.0))

# # Execution time testing (100 runs for each sigma)
# print("\nTesting execution times (100 runs for each sigma value)...")
# print("-" * 60)

# # Pre-load and preprocess images once
# print("Pre-loading images for timing tests...")
# image1 = cv2.imread(file_low[3])
# image2 = cv2.imread(file_high[3])
# image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB) / 255.0
# image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2RGB) / 255.0
# image1 = cv2.resize(image1, (0, 0), fx=0.5, fy=0.5)
# image2 = cv2.resize(image2, (0, 0), fx=0.5, fy=0.5)

# all_execution_times = {sigma: [] for sigma in sigma_values}

# for sigma in sigma_values:
#     print(f"Testing σ = {sigma:.1f}...")
    
#     # Pre-compute filter for this sigma
#     cutoff_frequency = sigma
#     ksize = int(cutoff_frequency * 4 + 1)
#     filter = cv2.getGaussianKernel(ksize, cutoff_frequency)
#     filter = filter @ filter.T
    
#     for run in range(100):
#         start_time = time.time()
        
#         # Only measure the filtering operations
#         low_frequencies_gaussian  = my_imfilter(image1, filter)
#         high_frequencies_gaussian = image2 - my_imfilter(image2, filter)
#         hybrid_image_gaussian     = low_frequencies_gaussian + high_frequencies_gaussian
        
#         end_time = time.time()
#         execution_time = end_time - start_time
#         all_execution_times[sigma].append(execution_time)

# # Calculate statistics
# print("\nExecution Time Results (100 runs each):")
# print("-" * 60)
# for sigma in sigma_values:
#     times = all_execution_times[sigma]
#     avg_time = np.mean(times)
#     std_time = np.std(times)
#     min_time = np.min(times)
#     max_time = np.max(times)
    
#     print(f"σ = {sigma:.1f}:")
#     print(f"  Average: {avg_time:.6f} seconds")
#     print(f"  Std Dev: {std_time:.6f} seconds")
#     print(f"  Min:     {min_time:.6f} seconds")
#     print(f"  Max:     {max_time:.6f} seconds")
#     print()

# # Find fastest and slowest
# all_avg_times = [np.mean(all_execution_times[sigma]) for sigma in sigma_values]
# fastest_idx = np.argmin(all_avg_times)
# slowest_idx = np.argmax(all_avg_times)

# print(f"Fastest average: σ = {sigma_values[fastest_idx]:.1f} ({all_avg_times[fastest_idx]:.6f}s)")
# print(f"Slowest average: σ = {sigma_values[slowest_idx]:.1f} ({all_avg_times[slowest_idx]:.6f}s)")

# # Display images in three separate plots (3x5 layout)
# # Plot 1: Low Frequencies
# plt.figure(figsize=(20, 12))
# for i in range(15):
#     plt.subplot(3, 5, i+1)
#     plt.imshow(low_freq_images[i])
#     plt.title(f'Low Frequencies\nσ = {sigma_values[i]}', fontsize=10, pad=10)
#     plt.axis('off')
# plt.suptitle('Low Frequencies for Different Sigma Values', fontsize=16, y=0.95)
# plt.tight_layout()
# plt.subplots_adjust(top=0.9)
# plt.show()

# # Plot 2: High Frequencies
# plt.figure(figsize=(20, 12))
# for i in range(15):
#     plt.subplot(3, 5, i+1)
#     plt.imshow(high_freq_images[i])
#     plt.title(f'High Frequencies (+0.5)\nσ = {sigma_values[i]}', fontsize=10, pad=10)
#     plt.axis('off')
# plt.suptitle('High Frequencies for Different Sigma Values', fontsize=16, y=0.95)
# plt.tight_layout()
# plt.subplots_adjust(top=0.9)
# plt.show()

# # Plot 3: Hybrid Images
# plt.figure(figsize=(20, 12))
# for i in range(15):
#     plt.subplot(3, 5, i+1)
#     plt.imshow(hybrid_images[i])
#     plt.title(f'Hybrid Image\nσ = {sigma_values[i]}', fontsize=10, pad=10)
#     plt.axis('off')
# plt.suptitle('Hybrid Images for Different Sigma Values', fontsize=16, y=0.95)
# plt.tight_layout()
# plt.subplots_adjust(top=0.9)
# plt.show()

# # Create execution time visualization
# avg_times = [np.mean(all_execution_times[sigma]) for sigma in sigma_values]
# std_times = [np.std(all_execution_times[sigma]) for sigma in sigma_values]

# plt.figure(figsize=(12, 5))

# # Bar chart with error bars
# plt.subplot(1, 2, 1)
# bars = plt.bar(range(len(sigma_values)), avg_times, yerr=std_times, 
#                color='skyblue', edgecolor='navy', alpha=0.7, capsize=5)
# plt.xlabel('Sigma Values')
# plt.ylabel('Average Execution Time (seconds)')
# plt.title('Average Execution Time vs Sigma Values\n(Error bars show standard deviation)')
# plt.xticks(range(len(sigma_values)), [f'σ = {s:.1f}' for s in sigma_values])
# plt.grid(True, alpha=0.3)

# # Add value labels on bars
# for i, (avg, std) in enumerate(zip(avg_times, std_times)):
#     plt.text(i, avg + std + 0.0001, f'{avg:.6f}s', ha='center', va='bottom', fontsize=8)

# # Line plot
# plt.subplot(1, 2, 2)
# plt.errorbar(sigma_values, avg_times, yerr=std_times, 
#              marker='o', linewidth=2, markersize=8, color='red', 
#              markerfacecolor='lightcoral', capsize=5)
# plt.xlabel('Sigma Values')
# plt.ylabel('Average Execution Time (seconds)')
# plt.title('Execution Time Trend\n(Error bars show standard deviation)')
# plt.grid(True, alpha=0.3)

# # Add value labels on points
# for i, (avg, std) in enumerate(zip(avg_times, std_times)):
#     plt.text(sigma_values[i], avg + std + 0.0001, f'{avg:.6f}s', ha='center', va='bottom', fontsize=8)

# plt.tight_layout()
# plt.show()
