# Grayscale (single method, no parameters): luminosity (BT.601)
import numpy as np
from common import to_float01, to_uint8, load_image, data_path
import matplotlib.pyplot as plt

def grayscale_average(img):
    """Return 3-channel grayscale using Y = 0.299R + 0.587G + 0.114B."""
    f = to_float01(img)
    R, G, B = f[...,0], f[...,1], f[...,2]
    ### Your Code ###
    Y = (R + G + B) / 3
    Y3 = np.stack([Y, Y, Y], axis=-1)
    return to_uint8(Y3)

if __name__ == "__main__":
    img = load_image(data_path("image.jpg"))
    img_gray = grayscale_average(img)
    plt.imshow(img_gray)
    plt.show()
