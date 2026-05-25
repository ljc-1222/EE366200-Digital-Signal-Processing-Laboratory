# Horizontal / Vertical / Both flips using indexing only
from common import load_image, data_path
import matplotlib.pyplot as plt

def flip_image(img, mode='horizontal'):
    if mode == 'horizontal':
        ### Your Code ###
        img_horizantal = img[:, ::-1, :]
        return img_horizantal
    elif mode == 'vertical':
        ### Your Code ###
        img_vertical = img[::-1, :, :]
        return img_vertical
    elif mode == 'both':
        ### Your Code ###
        img_both = img[::-1, ::-1, :]
        return img_both
    else:
        raise ValueError("mode must be 'horizontal', 'vertical', or 'both'")

if __name__ == "__main__":
    img = load_image(data_path("image.jpg"))
    img_horizontal = flip_image(img, 'horizontal')
    img_vertical = flip_image(img, 'vertical')
    img_both = flip_image(img, 'both')
    plt.figure()
    plt.title("Horizontal")
    plt.imshow(img_horizontal)
    plt.show(block=False)
    plt.figure()
    plt.title("Vertical")
    plt.imshow(img_vertical)
    plt.show(block=False)
    plt.figure()
    plt.title("Both")
    plt.imshow(img_both)
    plt.show()
