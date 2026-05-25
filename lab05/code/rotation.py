import numpy as np
from common import to_float01, to_uint8, load_image, data_path
import matplotlib.pyplot as plt

def rotate_word_style(img, radius_rad, bg_val=0.0):

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

    # --- backward warping using bilinear interpolation
    # For each (x_new, y_new) find the corresponding (x_old, y_old)
    # hint：array index is [row=y, col=x]
    cm, sm = np.cos(-radius_rad), np.sin(-radius_rad)  # 逆旋轉
    for y_new in range(height_new):
        for x_new in range(width_new):
            # step5: shift the new pixel (y_new, x_new) back, and rotate -radius
            # degree to get (y_old, x_old)
            ### Your Code ###
            y_old = (x_new - x_shift - 1) * sm + (y_new - y_shift - 1) * cm
            x_old = (x_new - x_shift - 1) * cm - (y_new - y_shift - 1) * sm

            # step6: use "ceil()" & "floor()" to get interpolation coordinates
            ### Your Code ###
            y1 = np.ceil(y_old).astype(int)
            y2 = np.floor(y_old).astype(int)
            x1 = np.ceil(x_old).astype(int)
            x2 = np.floor(x_old).astype(int)

            # step7:
            if (1 <= x1 <= w) and (1 <= x2 <= w) and (1 <= y1 <= h) and (1 <= y2 <= h):
                # step8: weight wa, wb, notice that if x1 = x2 or y1 = y2,
                # function "wa = ()/(x1-x2)" will be fail.
                # at this situation, you need to assign a value to wa directly.
                ### Your Code ###
                wa = (x_old - x2) / (x1 - x2) if x1 != x2 else 1
                wb = (y_old - y2) / (y1 - y2) if y1 != y2 else 1

                # step9: calculate weight w1, w2, w3, w4.
                ### Your Code ###
                w1 = (1 - wa) * (1 - wb)
                w2 = wa * (1 - wb)
                w3 = wa * wb
                w4 = (1 - wa) * wb

                # Move back to 0-based
                x1z, x2z = x1 - 1, x2 - 1
                y1z, y2z = y1 - 1, y2 - 1

                Q11 = f[y1z, x1z]
                Q21 = f[y1z, x2z]
                Q22 = f[y2z, x2z]
                Q12 = f[y2z, x1z]
                # step10: calculate new [x,y] with 4 neighbor points and their weights
                ### Your Code ###
                out[y_new, x_new] = w1 * Q11 + w2 * Q21 + w3 * Q22 + w4 * Q12
            else:
                out[y_new, x_new] = bg_val  # 超界 → 背景
    return to_uint8(out)

if __name__ == "__main__":
    img = load_image(data_path("image.jpg"))
    img_rotation = rotate_word_style(img, radius_rad=np.pi/4)
    plt.figure()
    plt.title("Radius = 45°")
    plt.imshow(img_rotation)
    plt.show(block=False)
    
    img_rotation = rotate_word_style(img, radius_rad=np.pi/2)
    plt.figure()
    plt.title("Radius = 90°")
    plt.imshow(img_rotation)
    plt.show(block=False)
    
    img_rotation = rotate_word_style(img, radius_rad=np.pi)
    plt.figure()
    plt.title("Radius = 180°")
    plt.imshow(img_rotation)
    plt.show(block=False)
    
    img_rotation = rotate_word_style(img, radius_rad=5*np.pi/4)
    plt.figure()
    plt.title("Radius = 225°")
    plt.imshow(img_rotation)
    plt.show(block=False)
    
    img_rotation = rotate_word_style(img, radius_rad=3*np.pi/2)
    plt.figure()
    plt.title("Radius = 270°")
    plt.imshow(img_rotation)
    plt.show(block=False)
    
    img_rotation = rotate_word_style(img, radius_rad=7*np.pi/4)
    plt.figure()
    plt.title("Radius = 315°")
    plt.imshow(img_rotation)
    plt.show(block=False)
    
    img_rotation = rotate_word_style(img, radius_rad=2*np.pi)
    plt.figure()
    plt.title("Radius = 360°")
    plt.imshow(img_rotation)
    plt.show()
