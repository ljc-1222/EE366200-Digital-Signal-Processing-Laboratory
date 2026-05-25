import numpy as np
from common import to_float01, to_uint8, load_image, data_path
import matplotlib.pyplot as plt

def rotate_word_style_forward(img, radius_rad, bg_val=0.0):

    h, w, _ = img.shape
    f = to_float01(img)  # [0,1] float

    # --- step1: record image vertices, and use rotation matrix to get new vertices (1,1) as origin

    # hint:("@"" is matrix multiplication operator,
    # "*" is elelement-wise multiplication operator,
    # .T" is transpose)
    
    rotatoin_matrix = np.array([[np.cos(radius_rad), -np.sin(radius_rad)], [np.sin(radius_rad), np.cos(radius_rad)]])
    vertex = np.array([[0, 0], [w - 1, 0], [0, h - 1], [w - 1, h - 1]])
    vertex_new = vertex @ rotatoin_matrix.T
     
    # --- step2: Find min x, min y, max x, max y, use "min()" & "max()" function is ok
    ###
    min_x, max_x = vertex_new[:, 0].min(), vertex_new[:, 0].max()
    min_y, max_y = vertex_new[:, 1].min(), vertex_new[:, 1].max()

    # --- step3: Consider how much to shift the image to the positive axis
    # "ceil()" & "floor()" may help
    ###
    x_shift = int(np.ceil(abs(min_x)))
    y_shift = int(np.ceil(abs(min_y)))

    # --- step4: calculate new width and height and transform into integer
    width_new  = int(np.ceil(max_x) - np.floor(min_x))
    height_new = int(np.ceil(max_y) - np.floor(min_y))

    # --- 預備輸出（float）
    out = np.full((height_new, width_new, 3), fill_value=bg_val, dtype=np.float32)

    # --- forward warping using bilinear interpolation
    # For each (x_old, y_old) find the corresponding (x_new, y_new)
    # hint：array index is [row=y, col=x]
    cm, sm = np.cos(radius_rad), np.sin(radius_rad)
    for y_old in range(h):
        for x_old in range(w):
            x_new = ((x_old * cm - y_old * sm) + x_shift - 1)
            y_new = ((x_old * sm + y_old * cm) + y_shift - 1)
            # If x_new and y_new are not integers, ceil and floor to get the nearest integers
            x_new = int(np.ceil(x_new))
            y_new = int(np.ceil(y_new))
            if (1 <= x_new <= width_new) and (1 <= y_new <= height_new):
                out[y_new, x_new] = f[y_old, x_old]
            else:
                out[y_new, x_new] = bg_val
    return to_uint8(out)

if __name__ == "__main__":
    img = load_image(data_path("image.jpg"))
    img_rotation = rotate_word_style_forward(img, radius_rad=np.pi/4)
    plt.figure()
    plt.title("Radius = 45°")
    plt.imshow(img_rotation)
    plt.show(block=False)
    
    img_rotation = rotate_word_style_forward(img, radius_rad=np.pi/2)
    plt.figure()
    plt.title("Radius = 90°")
    plt.imshow(img_rotation)
    plt.show(block=False)
    
    img_rotation = rotate_word_style_forward(img, radius_rad=np.pi)
    plt.figure()
    plt.title("Radius = 180°")
    plt.imshow(img_rotation)
    plt.show(block=False)
    
    img_rotation = rotate_word_style_forward(img, radius_rad=5*np.pi/4)
    plt.figure()
    plt.title("Radius = 225°")
    plt.imshow(img_rotation)
    plt.show(block=False)
    
    img_rotation = rotate_word_style_forward(img, radius_rad=3*np.pi/2)
    plt.figure()
    plt.title("Radius = 270°")
    plt.imshow(img_rotation)
    plt.show(block=False)
    
    img_rotation = rotate_word_style_forward(img, radius_rad=7*np.pi/4)
    plt.figure()
    plt.title("Radius = 315°")
    plt.imshow(img_rotation)
    plt.show(block=False)
    
    img_rotation = rotate_word_style_forward(img, radius_rad=2*np.pi)
    plt.figure()
    plt.title("Radius = 360°")
    plt.imshow(img_rotation)
    plt.show()
