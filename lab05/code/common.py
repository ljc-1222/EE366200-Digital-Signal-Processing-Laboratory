# Common helpers shared across modules (I/O + core math)
import os
import numpy as np
try:
    import imageio.v3 as iio
except Exception:
    import imageio as iio

def load_image(path):
    img = iio.imread(path)
    if img.ndim == 2:
        # expand gray to 3 channels for a consistent HxWx3 pipeline
        img = np.stack([img, img, img], axis=-1)
    if img.shape[-1] == 4:
        img = img[:, :, :3]
    return np.ascontiguousarray(img.astype(np.uint8))

def save_image(path, img):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    img_to_write = np.clip(img, 0, 255).astype(np.uint8)
    iio.imwrite(path, img_to_write)

def data_path(filename):
    return os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", filename)

def to_float01(img_u8):
    return img_u8.astype(np.float32) / 255.0

def to_uint8(img_f):
    return np.clip(np.round(img_f * 255.0), 0, 255).astype(np.uint8)

def bilinear_sample(img_f, x, y, bg_val=0.0):
    """
    Bilinear sampling at (x,y) in float coords.
    Returns a 3-vector (RGB). Out-of-bounds -> bg_val.
    """
    h, w, c = img_f.shape
    if x < 0 or x > w - 1 or y < 0 or y > h - 1:
        return np.array([bg_val, bg_val, bg_val], dtype=np.float32)

    x0 = int(np.floor(x)); y0 = int(np.floor(y))
    x1 = min(x0 + 1, w - 1); y1 = min(y0 + 1, h - 1)
    dx = x - x0; dy = y - y0

    Q11 = img_f[y0, x0]
    Q21 = img_f[y0, x1]
    Q12 = img_f[y1, x0]
    Q22 = img_f[y1, x1]

    # element-wise linear blends along x then y (operates on 3 channels at once)
    top = (1 - dx) * Q11 + dx * Q21
    bot = (1 - dx) * Q12 + dx * Q22
    return (1 - dy) * top + dy * bot

def _transform_corners(h, w, A, c_src):
    """
    Transform the 4 corners with a 2x2 linear map A, around center c_src.
    Returns Nx2 array of transformed coordinates (no re-centering applied).
    """
    corners = np.array([[0, 0],
                        [w - 1, 0],
                        [0, h - 1],
                        [w - 1, h - 1]], dtype=np.float32)
    shifted = corners - c_src
    # '@' = matrix multiplication (matmul); '.T' = transpose
    transformed = shifted @ A.T
    return transformed

def warp_affine_centered(img, A, out_size=None, bg_val=0.0):
    """
    Centered affine warp with bilinear sampling.
    A: 2x2 linear transform (scale/rot/shear)
    out_size: (out_h, out_w) or None to auto-compute tight bbox
    """
    h, w, _ = img.shape
    c_src = np.array([(w - 1) / 2.0, (h - 1) / 2.0], dtype=np.float32)

    if out_size is None:
        t_corners = _transform_corners(h, w, A, c_src)
        xs, ys = t_corners[:, 0], t_corners[:, 1]
        out_w = int(np.floor(xs.max() - xs.min() + 1))
        out_h = int(np.floor(ys.max() - ys.min() + 1))
    else:
        out_h, out_w = out_size

    c_dst = np.array([(out_w - 1) / 2.0, (out_h - 1) / 2.0], dtype=np.float32)

    src_f = to_float01(img)
    out_f = np.zeros((out_h, out_w, 3), dtype=np.float32)
    A_inv = np.linalg.inv(A)

    for yy in range(out_h):
        yd = yy - c_dst[1]
        for xx in range(out_w):
            xd = xx - c_dst[0]
            # map dst -> src by A_inv, then re-center
            xs, ys = (A_inv @ np.array([xd, yd], dtype=np.float32))
            xs += c_src[0]; ys += c_src[1]
            out_f[yy, xx] = bilinear_sample(src_f, float(xs), float(ys), bg_val=bg_val)

    return to_uint8(out_f)
