import cv2
import numpy as np
import mediapipe as mp
import os
import importlib
import urllib.request

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LAB_DIR = os.path.dirname(BASE_DIR)
DATA_DIR = os.path.join(LAB_DIR, "data")
RESULTS_DIR = os.path.join(LAB_DIR, "results")
FACE_LANDMARKER_MODEL = os.path.join(DATA_DIR, "face_landmarker.task")
FACE_LANDMARKER_URL = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task"

# Load images
image_1 = cv2.imread(os.path.join(DATA_DIR, "person_1.jpg")).astype(np.float32) / 255.0
image_2 = cv2.imread(os.path.join(DATA_DIR, "person_2.jpg")).astype(np.float32) / 255.0
# image_1 = cv2.imread(os.path.join(DATA_DIR, "person_3.jpg")).astype(np.float32) / 255.0
# image_2 = cv2.imread(os.path.join(DATA_DIR, "person_4.jpg")).astype(np.float32) / 255.0

print(f"Input size: {image_1.shape}, {image_2.shape}")

def ensure_face_landmarker_model():
    """Download the MediaPipe Tasks face landmarker model if it is missing."""
    if not os.path.exists(FACE_LANDMARKER_MODEL):
        print(f"Downloading {os.path.basename(FACE_LANDMARKER_MODEL)}...")
        urllib.request.urlretrieve(FACE_LANDMARKER_URL, FACE_LANDMARKER_MODEL)
    return FACE_LANDMARKER_MODEL

def detect_landmarks(img):
    """Detect 468 face landmarks using MediaPipe."""
    h, w = img.shape[:2]
    img_rgb = (np.clip(img * 255.0, 0, 255)).astype(np.uint8)

    if hasattr(mp, "solutions"):
        mp_face_mesh = mp.solutions.face_mesh
        with mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5) as face_mesh:
            results = face_mesh.process(img_rgb)
            if not results.multi_face_landmarks:
                raise RuntimeError("No face landmarks detected.")
            lms = results.multi_face_landmarks[0].landmark
            return np.array([[lm.x * w, lm.y * h] for lm in lms], dtype=np.float32)

    vision = importlib.import_module("mediapipe.tasks.python.vision")
    model_path = ensure_face_landmarker_model()
    options = vision.FaceLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=model_path),
        running_mode=vision.RunningMode.IMAGE,
        num_faces=1,
    )
    with vision.FaceLandmarker.create_from_options(options) as landmarker:
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
        results = landmarker.detect(mp_image)
        if not results.face_landmarks:
            raise RuntimeError("No face landmarks detected.")
        lms = results.face_landmarks[0]
        return np.array([[lm.x * w, lm.y * h] for lm in lms], dtype=np.float32)

def select_keypoints(landmarks):
    """Select 5 stable keypoints: left eye, right eye, nose tip, mouth left, mouth right."""
    idx = [33, 263, 1, 61, 291]
    return landmarks[idx, :].astype(np.float32)

def flip_points(pts, width):
    """Horizontally flip 2D points."""
    flipped = pts.copy()
    flipped[:, 0] = (width - 1) - flipped[:, 0]
    return flipped

def estimate_affine(src, dst):
    """Estimate affine transform using RANSAC."""
    M, _ = cv2.estimateAffine2D(src, dst, method=cv2.RANSAC, ransacReprojThreshold=3.0, maxIters=2000, confidence=0.99, refineIters=50)
    return M

def warp_with_mask(img, M, out_w, out_h):
    """Warp image and return mask."""
    warped = cv2.warpAffine(img, M, (out_w, out_h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
    mask_src = np.ones((img.shape[0], img.shape[1]), dtype=np.uint8) * 255
    mask_warped = cv2.warpAffine(mask_src, M, (out_w, out_h), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return warped, (mask_warped > 0).astype(np.uint8)

def intersection_rect(mask_a, mask_b):
    """Compute the largest axis-aligned rectangle fully inside the intersection mask."""
    inter = (mask_a.astype(np.uint8) & mask_b.astype(np.uint8))

    # If no overlap, return a degenerate rect
    if inter.max() == 0:
        return 0, 0, 0, 0

    # A tiny erosion removes 1-pixel slivers at the boundary to avoid including padding wedges
    inter = cv2.erode(inter, np.ones((3, 3), np.uint8), iterations=1)

    h, w = inter.shape
    # Build "heights" histogram for each row (maximal rectangle in binary matrix)
    heights = np.zeros((h, w), dtype=np.int32)
    heights[0, :] = (inter[0, :] > 0).astype(np.int32)
    for i in range(1, h):
        heights[i, :] = ((inter[i, :] > 0).astype(np.int32)) * (heights[i - 1, :] + 1)

    best_area, best_rect = 0, (0, 0, 0, 0)
    for i in range(h):
        stack = []
        j = 0
        while j <= w:
            curr = heights[i, j] if j < w else 0
            if not stack or curr >= heights[i, stack[-1]]:
                stack.append(j)
                j += 1
            else:
                top = stack.pop()
                width = j if not stack else (j - stack[-1] - 1)
                height = heights[i, top]
                if height == 0:
                    continue
                area = width * height
                if area > best_area:
                    x = (stack[-1] + 1) if stack else 0
                    y = i - height + 1
                    best_area = area
                    best_rect = (x, y, width, height)
    return best_rect

def crop_by_rect(img, rect):
    """Crop image by rectangle."""
    x, y, w, h = rect
    return img[y:y+h, x:x+w]

def try_align(image_ref, image_mov, kp_ref, kp_mov):
    """Try alignment with given keypoints."""
    h_ref, w_ref = image_ref.shape[:2]
    M = estimate_affine(kp_mov, kp_ref)
    warped_mov, mask_mov = warp_with_mask(image_mov, M, w_ref, h_ref)
    mask_ref = np.ones((h_ref, w_ref), dtype=np.uint8)
    rect = intersection_rect(mask_ref, mask_mov)
    x, y, w, h = rect
    return {
        'M': M,
        'rect': rect,
        'area': w * h,
        'aligned_moving': crop_by_rect(warped_mov, rect),
        'aligned_reference': crop_by_rect(image_ref, rect)
    }

def align_faces_max_overlap(image_ref, image_mov, allow_flip=True):
    """Main alignment pipeline."""
    # Detect landmarks and select keypoints
    lm_ref = detect_landmarks(image_ref)
    lm_mov = detect_landmarks(image_mov)
    kp_ref = select_keypoints(lm_ref)
    kp_mov = select_keypoints(lm_mov)

    # Try without flip
    result_a = try_align(image_ref, image_mov, kp_ref, kp_mov)
    result_a['flipped'] = False

    # Try with flip
    result_b = None
    if allow_flip:
        h2, w2 = image_mov.shape[:2]
        kp_mov_flip = flip_points(kp_mov, w2)
        M_flip = estimate_affine(kp_mov_flip, kp_ref)
        img_mov_flipped = cv2.flip(image_mov, 1)
        h_ref, w_ref = image_ref.shape[:2]
        warped_mov_flip, mask_mov_flip = warp_with_mask(img_mov_flipped, M_flip, w_ref, h_ref)
        mask_ref = np.ones((h_ref, w_ref), dtype=np.uint8)
        rect_flip = intersection_rect(mask_ref, mask_mov_flip)
        x, y, w, h = rect_flip
        result_b = {
            'M': M_flip,
            'rect': rect_flip,
            'area': w * h,
            'aligned_moving': crop_by_rect(warped_mov_flip, rect_flip),
            'aligned_reference': crop_by_rect(image_ref, rect_flip),
            'flipped': True
        }

    # Choose best result by overlap area
    candidates = [result_a]
    if result_b:
        candidates.append(result_b)
    best = max(candidates, key=lambda d: d['area'])

    return {
        'aligned_ref': best['aligned_reference'],
        'aligned_mov': best['aligned_moving'],
        'M': best['M'],
        'flipped': best['flipped'],
        'crop_rect': best['rect'],
        'overlap_area': int(best['area'])
    }

# Run alignment
result = align_faces_max_overlap(image_ref=image_1, image_mov=image_2, allow_flip=True)
out_ref = result['aligned_ref']
out_mov = result['aligned_mov']
print(f"Output size: {out_ref.shape}, {out_mov.shape}")

# Save outputs
os.makedirs(RESULTS_DIR, exist_ok=True)
cv2.imwrite(os.path.join(RESULTS_DIR, "aligned_1.jpg"), (np.clip(out_ref * 255.0, 0, 255)).astype(np.uint8))
cv2.imwrite(os.path.join(RESULTS_DIR, "aligned_2.jpg"), (np.clip(out_mov * 255.0, 0, 255)).astype(np.uint8))
