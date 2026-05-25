# EE366200 Digital Signal Processing Laboratory

## Course Information

- **Course:** EE366200 Digital Signal Processing Laboratory
- **Language:** Python
- **Term:** 2025 Autumn
- **Repository scope:** lab01-lab08 source code, provided lab handouts, input data, submitted reports, and selected generated results.

This repository collects the programming assignments for the DSP Laboratory course. The first half focuses on audio signal processing, including STFT, Mel filter banks, MFCC, audio reconstruction, and audio classification. The second half focuses on image signal processing, including pixel transforms, filtering, hybrid images, Harris corner detection, and seam carving.

## Repository Layout

| Path | Topic | Main files |
| --- | --- | --- |
| `lab01/` | MFCC front-end basics: pre-emphasis, STFT, Mel/Hz conversion, Mel filter banks, log compression, and DCT. | `Lab1_student.py`, `Lab1_functions_student.py`, `audio.wav`, reports |
| `lab02/` | Inverse MFCC and audio reconstruction: inverse DCT, Mel-bank pseudo-inverse, Griffin-Lim, de-emphasis, and SNR experiments. | `Lab2_student.py`, `Lab2_functions_student.py`, `Lab2_stft2audio_student.py`, `audio.wav`, reconstructed audio |
| `lab03/` | Environmental sound classification using handcrafted audio features and classical ML models. | `Lab3_111061220.py`, `Lab3_111061220_functions.py`, `data/`, report |
| `lab04/` | Baby sound classification challenge with feature caching, audio augmentation, robust scaling, and ensemble gradient boosting. | `Lab4_111061220.py`, `Lab4_111061220_functions.py`, `2025-dsp-lab-detecting-baby-sounds/`, result CSV |
| `lab05/` | Basic image operations and geometric transforms: grayscale, flipping, resizing, rotation, shear, nearest-neighbor and bilinear interpolation. | `code/`, `data/`, `results/`, report |
| `lab06/` | Image filtering and hybrid images: custom convolution, Gaussian/box/Laplacian filters, high-pass and low-pass combinations, face alignment helper. | `code/`, `data/`, `results/`, report |
| `lab07/` | Harris corner detection and shift estimation from detected feature points. | `code/FindCorners.py`, `code/MyHarrisCornerDetector.py`, `code/CalculateShift.py`, `data/`, results |
| `lab08/` | Seam carving: energy maps, dynamic-programming seam search, seam reduction/insertion, object removal, and content amplification. | `code/`, `data/sea.jpg`, `results/`, report |

## Requirements

- Python 3.10 or newer is recommended.
- A POSIX-like shell is useful for the example commands below.
- No dependency lock file is provided. Install the packages required by the labs you want to run.

Common packages used across the labs:

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install numpy scipy matplotlib librosa soundfile pandas scikit-learn tqdm opencv-python scikit-image imageio
```

`lab06/code/alignment.py` additionally requires MediaPipe:

```sh
python -m pip install mediapipe
```

When `lab06/code/alignment.py` is run with the newer MediaPipe Tasks API, it downloads `face_landmarker.task` into `lab06/data/` if the model file is missing.

## Run Examples

Most scripts can be run directly from the repository root because file paths are resolved relative to each script location:

```sh
python lab01/Lab1_student.py
python lab02/Lab2_student.py
python lab03/Lab3_111061220.py
python lab04/Lab4_111061220.py
```

For image-processing labs:

```sh
python lab05/code/main.py
python lab06/code/test_my_imfilter.py
python lab06/code/generate_hybrid_image.py
python lab07/code/MyHarrisCornerDetector.py
python lab08/code/seamCarvingTester.py
```

Some scripts open Matplotlib or OpenCV windows. Close the displayed window to continue execution. `lab08/code/seamCarvingTester.py` includes an interactive object-removal step that uses mouse drawing in an OpenCV window.

## Data and Outputs

- `lab01/audio.wav` and `lab02/audio.wav` are the source audio files for MFCC and inverse-MFCC experiments.
- `lab03/data/` contains the environmental-sound dataset layout used by `Lab3_111061220.py`.
- `lab04/2025-dsp-lab-detecting-baby-sounds/Baby_Data/` contains the baby-sound challenge data used by `Lab4_111061220.py`.
- `lab05/results/`, `lab06/results/`, `lab07/results/`, and `lab08/results/` store generated figures or output images.
- Reports are stored as `*.pdf` or `*.docx` files in each lab directory.

## File Naming Notes

- `*_student.py` files are the lab starter/submission scripts.
- `*_functions*.py` files contain helper functions shared by the corresponding lab.
- `code/` contains modular image-processing implementations for lab05-lab08.
- `data/` contains input signals, images, or challenge datasets.
- `results/` contains generated images, figures, or submission outputs.
- `report/`, `reports/`, and `*_report_*.pdf` contain submitted lab reports.

## Course Scope Note

The course also includes lab09-lab12. Those labs are related to biomedical signal measurement and depend heavily on external circuit-board components, wiring, and measurement hardware, so the corresponding files and experiment results are not included in this repository.
