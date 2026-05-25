# Zero-parameter interactive runner: close each window to proceed.
import os
import numpy as np
from common import load_image, save_image
from grayscale import grayscale
from flip import flip_image
from rotation import rotate_word_style
from shear import shear_image
from resize import resize_image
from grayscale_average import grayscale_average

# ==== Configuration (edit here if needed) ====
DEFAULT_INPUT = "image.jpg"   # file under LAB5/data/
Flip_Mode = "horizontal"
ROTATE_RAD = np.pi / 6
SHEAR_SH_X    = 0.5
SHEAR_SH_Y    = 0.0
RESIZE_SCALE  = 0.6
# ============================================

# Optional viewer
try:
    import matplotlib.pyplot as plt
    _HAS_PLT = True
except Exception:
    _HAS_PLT = False

def _io_paths(input_name):
    base = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(os.path.dirname(base), "data")
    out_dir  = os.path.join(os.path.dirname(base), "results")
    os.makedirs(out_dir, exist_ok=True)
    src_path = os.path.join(data_dir, input_name)
    return src_path, out_dir

def _show_single(title, img):
    if not _HAS_PLT:
        print("[INFO] matplotlib not installed; cannot show windows. Install via `pip install matplotlib`.")
        return
    plt.figure(figsize=(6,5))
    plt.title(title + " — close to continue")
    plt.imshow(img)
    plt.axis('off')
    plt.show(block=True)

def main():
    src_path, out_dir = _io_paths(DEFAULT_INPUT)
    img = load_image(src_path)

    steps = [
        ("Gray", lambda: grayscale(img), "gray.png"),
        (f"Flip {Flip_Mode}",   lambda: flip_image(img, Flip_Mode), "flip_image.png"),
        ("Rotate (word-style)", lambda: rotate_word_style(img, radius_rad=ROTATE_RAD), "rotate_image.png"),
        (f"Shear sh_x={SHEAR_SH_X}", lambda: shear_image(img, sh_x=SHEAR_SH_X, sh_y=SHEAR_SH_Y), "shear_image.png"),
        (f"Scale {RESIZE_SCALE}x", lambda: resize_image(img, scale=RESIZE_SCALE), f"scale_{RESIZE_SCALE}x.png"),
    ]

    for i, (title, fn, fname) in enumerate(steps, start=1):
        arr = fn()
        save_image(os.path.join(out_dir, fname), arr)
        _show_single(f"[{i}/{len(steps)}] {title}", arr)

    print("[DONE] Interactive demo finished. Images saved under results/.")

if __name__ == "__main__":
    main()
