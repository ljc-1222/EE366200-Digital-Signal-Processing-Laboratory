import numpy as np
import cv2
from my_imfilter import my_imfilter
import matplotlib.pyplot as plt
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(os.path.dirname(BASE_DIR), "data")

## Setup
test_image = cv2.imread(os.path.join(DATA_DIR, 'cat.jpg'))
test_image = cv2.cvtColor(test_image, cv2.COLOR_BGR2RGB) / 255.0
test_image = cv2.resize(test_image, (0, 0), fx=0.7, fy=0.7, interpolation=cv2.INTER_LINEAR)

plt.figure(1)
plt.imshow(test_image)
plt.title('Original Image')

## Identity filter
identity_filter = np.array([[0, 0, 0],
                            [0, 1, 0],
                            [0, 0, 0]], dtype=np.float32)
identity_image = my_imfilter(test_image, identity_filter)

plt.figure(2)
plt.imshow(identity_image)
plt.title('Identity Filter Result')

# Converts the original RGB image to grayscale for visual reference.
gray_image = cv2.cvtColor((test_image * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
gray_image = gray_image.astype(np.float32) / 255.0

plt.figure(3)
plt.imshow(gray_image, cmap='gray')
plt.title('Grayscale Image')

## Small blur filter (Box Blur)
blur_filter = np.ones((3, 3), dtype=np.float32)
blur_filter /= np.sum(blur_filter)
blur_image = my_imfilter(test_image, blur_filter)

plt.figure(4)
plt.imshow(blur_image)
plt.title('Box Blur Filter Result')

## Large Gaussian blur (separable)
large_1d_gauss = cv2.getGaussianKernel(25, 10).astype(np.float32)
large_blur_image = my_imfilter(test_image, large_1d_gauss)
large_blur_image = my_imfilter(large_blur_image, large_1d_gauss.T)

plt.figure(5)
plt.imshow(large_blur_image)
plt.title('Large Gaussian Blur (1D separable)')

## Sobel filter
sobel_filter = np.array([[-1, 0, 1],
                         [-2, 0, 2],
                         [-1, 0, 1]], dtype=np.float32)
sobel_image = my_imfilter(test_image, sobel_filter)

plt.figure(6)
plt.imshow(np.clip(sobel_image + 0.5, 0.0, 1.0))
plt.title('Sobel Filter Result (+0.5 shift)')

## Laplacian filter
laplacian_filter = np.array([[0, 1, 0],
                             [1, -4, 1],
                             [0, 1, 0]], dtype=np.float32)
laplacian_image = my_imfilter(test_image, laplacian_filter)

plt.figure(7)
plt.imshow(np.clip(laplacian_image + 0.5, 0.0, 1.0))
plt.title('Laplacian Filter Result (+0.5 shift)')

## High pass (image - blur)
high_pass_image = test_image - blur_image

plt.figure(8)
plt.imshow(np.clip(high_pass_image + 0.5, 0.0, 1.0))
plt.title('High-Pass Image (image - blur) (+0.5 shift)')

plt.show()
