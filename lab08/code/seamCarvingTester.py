import cv2
import numpy as np
from seamCarvingReduce import seamCarvingReduce
from seamCarvingInsert import seamCarvingInsert
from seamCarvingObjectRemove import seamCarvingObjectRemove
from seamCarvingContentAmplification import seamCarvingContentAmplification
import matplotlib.pyplot as plt
import os

# --- Load image ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LAB_DIR = os.path.dirname(BASE_DIR)
DATA_DIR = os.path.join(LAB_DIR, 'data')
RESULTS_DIR = os.path.join(LAB_DIR, 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)

img_path = os.path.join(DATA_DIR, 'sea.jpg')
if not os.path.exists(img_path):
    raise FileNotFoundError(f"{img_path} not found")

image = cv2.imread(img_path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
h, w = image.shape[:2]
image = cv2.resize(image, (w // 3, h // 3))
image_orig = image.copy()

# --- 1) Original ---
plt.figure(); plt.imshow(image_orig); plt.title("Original Image")

# --- 2) Scaling ---
image_scaling = cv2.resize(image_orig, (image_orig.shape[1] // 2, image_orig.shape[0]))
plt.figure(); plt.imshow(image_scaling); plt.title("Scaling")

# --- 3) Cropping ---
image_cropping = image_orig[:, :image_orig.shape[1] // 2, :].copy()
plt.figure(); plt.imshow(image_cropping); plt.title("Cropping")

# --- 4) Seam Carving Reduce ---
reduce_input = image_orig.copy()
reduce_amount = image_orig.shape[1] // 4
print(f"Running seamCarvingReduce: removing {reduce_amount} seams (vertical)...")
image_reduce = seamCarvingReduce(reduce_input, reduce_amount, 1)
try:
    cv2.imwrite(os.path.join(RESULTS_DIR, 'seamCarving_reduce_horizontal.jpg'), cv2.cvtColor(image_reduce, cv2.COLOR_RGB2BGR))
except Exception as e:
    print("Warning: could not save seamCarving_reduce_horizontal.jpg:", e)
plt.figure(); plt.imshow(image_reduce); plt.title("Seam Carving Reduce")

# --- 5) Seam Insertion ---
insert_input = image_orig.copy()
insert_amount = image_orig.shape[1] // 4
print(f"Running seamCarvingInsert: inserting {insert_amount} seams (vertical)...")
image_insert = seamCarvingInsert(insert_input, insert_amount)
try:
    cv2.imwrite(os.path.join(RESULTS_DIR, 'seamCarving_insert.jpg'), cv2.cvtColor(image_insert, cv2.COLOR_RGB2BGR))
except Exception as e:
    print("Warning: could not save seamCarving_insert.jpg:", e)
plt.figure(); plt.imshow(image_insert); plt.title("Seam Insertion")

# --- 6) Object Removal (OpenCV freehand) ---
print("\nObject removal:")
print(" - Left mouse button: press and drag to draw mask (freehand).")
print(" - Release mouse to stop a stroke; you can draw multiple strokes.")
print(" - Press 'r' to reset mask and start over.")
print(" - Press 'q' when done to run seam carving object removal.\n")

proc_img = image_orig.copy()
disp = proc_img.copy()
mask = np.zeros(proc_img.shape[:2], dtype=np.uint8)
drawing = False
ix, iy = -1, -1

def draw(event, x, y, flags, param):
    global drawing, ix, iy, disp, mask
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix, iy = x, y
    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            cv2.line(mask, (ix, iy), (x, y), 1, thickness=15)
            cv2.line(disp, (ix, iy), (x, y), (255, 0, 0), thickness=2)
            ix, iy = x, y
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False

win_name = "Draw mask - press q when done, r to reset"
cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
cv2.setMouseCallback(win_name, draw)

while True:
    cv2.imshow(win_name, cv2.cvtColor(disp, cv2.COLOR_RGB2BGR))
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('r'):
        mask[:] = 0
        disp = proc_img.copy()
        print("Mask reset")

cv2.destroyWindow(win_name)

if not np.any(mask):
    print("No mask drawn -> skipping object removal.")
    image_remove = proc_img.copy()
else:
    print("Running seamCarvingObjectRemove (this may take a while)...")
    image_remove = seamCarvingObjectRemove(proc_img.copy(), mask.astype(bool), 0)
    try:
        cv2.imwrite(os.path.join(RESULTS_DIR, 'seamCarving_object_remove_vertical.jpg'), cv2.cvtColor(image_remove, cv2.COLOR_RGB2BGR))
    except Exception as e:
        print("Warning: could not save seamCarving_object_remove_vertical.jpg:", e)
plt.figure(); plt.imshow(image_remove); plt.title("Object Removed")

# --- 7) Content Amplification ---
print("\nRunning Content Amplification (may take some time)...")
image_amp = seamCarvingContentAmplification(image_orig.copy())
try:
    cv2.imwrite(os.path.join(RESULTS_DIR, 'seamCarving_content_amplification.jpg'), cv2.cvtColor(image_amp, cv2.COLOR_RGB2BGR))
except Exception as e:
    print("Warning: could not save seamCarving_content_amplification.jpg:", e)
plt.figure(); plt.imshow(image_amp); plt.title("Content Amplification")

plt.show()
