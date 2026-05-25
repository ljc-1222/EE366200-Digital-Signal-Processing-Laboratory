import numpy as np
from common import to_float01, to_uint8, load_image, data_path
import matplotlib.pyplot as plt

def shear_image(img, sh_x, sh_y, bg_val=0.0):
    """
    input:
        img   : uint8 HxWx3
        sh_x  : x-axis shear amount (e.g., -0.8)
        sh_y  : y-axis shear amount (e.g.,  0.2)
        bg_val: out-of-bounds background value in [0,1]
    output:
        uint8 H'xW'x3
    """
    h, w, _ = img.shape
    f = to_float01(img)  # [0,1] float

    ## create new image
    # step 1. record image vertices, and use the transformation matrix to get new vertices.
    matrix = np.array([[1.0, sh_x], [sh_y, 1.0]], dtype=np.float32)
    matrix_inv = np.linalg.inv(matrix)
    vertex = np.array([[0, 0], [w - 1, 0], [0, h - 1], [w - 1, h - 1]])
    vertex_new = vertex @ matrix.T

    # step 2. find min x, min y, max x, max y, use "min()" & "max()" function is ok
    ### Your code ###
    min_x, max_x = vertex_new[:, 0].min(), vertex_new[:, 0].max()
    min_y, max_y = vertex_new[:, 1].min(), vertex_new[:, 1].max()

    # step3. consider how much to shift the image to the positive axis
    x_shift = int(np.ceil(abs(min_x)))
    y_shift = int(np.ceil(abs(min_y)))

    # step4. calculate new width and height, if they are not integer,  use
    # "ceil()" & "floor()" to help get the largest width and height.
    width_new = int(np.ceil(max_x) - np.floor(min_x))
    height_new = int(np.ceil(max_y) - np.floor(min_y))
 
    # step5 initial new image
    out = np.full((height_new, width_new, 3), fill_value=bg_val, dtype=np.float32)

    ## backward warping using bilinear interpolation
    # for each pixel on the shear image, map back, then bilinear from f (float in [0,1])
    for y_new in range(1, height_new + 1):     # 1-based loop to match lecture
        for x_new in range(1, width_new + 1):

            # step5. shift (y_new, x_new) back and reverse the shear transform to get (y_old, x_old)
            ### Your Code ###
            (x_old, y_old) = matrix_inv @ np.array([x_new - x_shift, y_new - y_shift], dtype=np.float32)

            # step6. ceil()/floor() to get interpolation coordinates x1, x2, y1, y2
            ### Your Code ###
            y1 = np.ceil(y_old).astype(int)
            y2 = np.floor(y_old).astype(int)
            x1 = np.ceil(x_old).astype(int)
            x2 = np.floor(x_old).astype(int)


            # step7. if (y_old, x_old) is inside of the source image, 
            # calculate r, g, b by interpolation.
            if (1 <= x1 <= w) and (1 <= x2 <= w) and (1 <= y1 <= h) and (1 <= y2 <= h):
                # step8. calculate weight wa, wb, notice that if x1 = x2 or y1 = y2,
                # function "wa = ()/(x1-x2)" will be fail. 
                # at this situation, you need to assign a value to wa directly.
                ### Your Code ###
                wa = (x_old - x2) / (x1 - x2) if x1 != x2 else 1
                wb = (y_old - y2) / (y1 - y2) if y1 != y2 else 1

                # step9. calculate weight w1, w2, w3, w4
                ### Your Code ###
                w1 = (1 - wa) * (1 - wb)
                w2 = wa * (1 - wb)
                w3 = wa * wb
                w4 = (1 - wa) * wb
                
                # 0-based indices for array access
                x1z, x2z = x1 - 1, x2 - 1
                y1z, y2z = y1 - 1, y2 - 1
                
                Q11 = f[y1z, x1z]
                Q21 = f[y1z, x2z]
                Q22 = f[y2z, x2z]
                Q12 = f[y2z, x1z]

                # step10. rgb by 4 neighbors + weights
                ### Your Code ###
                out[y_new - 1, x_new - 1] = w1 * Q11 + w2 * Q21 + w3 * Q22 + w4 * Q12
            else:
                out[y_new - 1, x_new - 1] = bg_val

    # write back (convert loop indices to 0-based)
    return to_uint8(out)

if __name__ == "__main__":
    img = load_image(data_path("image.jpg"))
    img_shear = shear_image(img, sh_x=0.2, sh_y=0.2)
    plt.figure()
    plt.title("Shear sh_x=0.2, sh_y=0.2")
    plt.imshow(img_shear)
    plt.show(block=False)
    
    img_shear = shear_image(img, sh_x=0.2, sh_y=2)
    plt.figure()
    plt.title("Shear sh_x=0.2, sh_y=2")
    plt.imshow(img_shear)
    plt.show(block=False)
    
    img_shear = shear_image(img, sh_x=2, sh_y=0.2)
    plt.figure()
    plt.title("Shear sh_x=2, sh_y=0.2")
    plt.imshow(img_shear)
    plt.show(block=False)
    
    img_shear = shear_image(img, sh_x=2, sh_y=2)
    plt.figure()
    plt.title("Shear sh_x=2, sh_y=2")
    plt.imshow(img_shear)
    plt.show()
