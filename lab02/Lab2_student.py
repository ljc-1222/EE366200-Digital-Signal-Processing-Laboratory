'''
@Modified by Paul Cho; 10th, Nov, 2020

For NTHU DSP Lab 2025 Autumn
'''

import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt
import os
from scipy.fftpack import dct
from scipy.fftpack import idct
from scipy.linalg import pinv
from scipy.optimize import nnls, lsq_linear
from scipy.signal import medfilt, medfilt2d

from Lab2_functions_student import pre_emphasis, de_emphasis, STFT, mel2hz, hz2mel, get_filter_banks
from Lab2_stft2audio_student import griffinlim
from Lab2_stft2audio_student import griffinlim_with_pghi_init as pghi

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
filename = os.path.join(BASE_DIR, 'audio.wav')
source_signal, sr = sf.read(filename) #sr:sampling rate
print('Sampling rate={} Hz.'.format(sr))

### hyper parameters
frame_length = 512                    # Frame length(samples)
frame_step = 256                      # Step length(samples)
emphasis_coeff = 0.95                 # pre-emphasis para
num_bands = 12                        # Filter number = band number
num_FFT = frame_length                # FFT freq-quantization
freq_min = 0
freq_max = int(0.5 * sr)
signal_length = len(source_signal)    # Signal length

# number of frames it takes to cover the entirety of the signal
num_frames = 1 + int(np.ceil((1.0 * signal_length - frame_length) / frame_step))

# spectrum of the source signal
original_spectrum = STFT(time_signal   = source_signal,
                         num_frames    = num_frames,
                         num_FFT       = num_FFT,
                         frame_step    = frame_step,
                         frame_length  = frame_length,
                         signal_length = signal_length,
                         verbose       = False)

# generation of pre_emphsized signal
pre_emphasized_signal = pre_emphasis(signal = source_signal, coefficient = emphasis_coeff)

# spectrum of the pre_emphasized signal
pre_emphasized_spectrum = STFT(time_signal   = pre_emphasized_signal,
                               num_frames    = num_frames,
                               num_FFT       = num_FFT,
                               frame_step    = frame_step,
                               frame_length  = frame_length,
                               signal_length = signal_length,
                               verbose       = False)

#YOUR CODE ENDS HERE;
##########################

'''
(1) Perform inverse DCT on MFCC (already done for you)
(2) Restore magnitude from logarithmic scale (i.e. use exponential)
(3) Invert the fbanks convolution
(4) Synthesize time-domain audio with Griffin-Lim
(5) Get STFT spectrogram of the reconstructed signal and compare it side by side with the original signal's STFT spectrogram
    (please convert magnitudes to logarithmic scale to better present the changes)
'''

# Build a MFCC result using num_bands = 12, 64
fbanks_12= get_filter_banks(12, num_FFT, sr, freq_min, freq_max)

filter_energies_12_original = np.dot(fbanks_12, original_spectrum)
features_12_original = np.log(filter_energies_12_original.T + np.finfo(float).eps)
MFCC_12_original = dct(features_12_original, norm = 'ortho')[:,:12]

filter_energies_12_pre_emphasized = np.dot(fbanks_12, pre_emphasized_spectrum)
features_12_pre_emphasized = np.log(filter_energies_12_pre_emphasized.T + np.finfo(float).eps)
MFCC_12_pre_emphasized = dct(features_12_pre_emphasized, norm = 'ortho')[:,:12]

fbanks_64 = get_filter_banks(64, num_FFT , sr, freq_min, freq_max)

filter_energies_64_original = np.dot(fbanks_64, original_spectrum)
features_64_original = np.log(filter_energies_64_original.T + np.finfo(float).eps)
MFCC_64_original = dct(features_64_original, norm = 'ortho')[:,:64]

filter_energies_64_pre_emphasized = np.dot(fbanks_64, pre_emphasized_spectrum)
features_64_pre_emphasized = np.log(filter_energies_64_pre_emphasized.T + np.finfo(float).eps)
MFCC_64_pre_emphasized = dct(features_64_pre_emphasized, norm = 'ortho')[:,:64]

# Random frame of MFCC (12 vs 64)
rand_idx = np.random.randint(0, MFCC_12_original.shape[0]) 
mfcc_frame_12_pre_emphasized = MFCC_12_pre_emphasized[rand_idx]
mfcc_frame_64_pre_emphasized = MFCC_64_pre_emphasized[rand_idx]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12,5))
fig.suptitle(' ')

ax1.plot(np.arange(len(mfcc_frame_12_pre_emphasized)), mfcc_frame_12_pre_emphasized)
ax1.set_title('MFCC of a random frame(12 banks)')
ax1.set_xlabel('Cepstral Coefficient')
ax1.set_ylabel('Magnitude')

ax2.plot(np.arange(len(mfcc_frame_64_pre_emphasized)), mfcc_frame_64_pre_emphasized)
ax2.set_title('MFCC of a random frame(64 banks)')
ax2.set_xlabel('Cepstral Coefficient')
ax2.set_ylabel('Magnitude')

plt.tight_layout()
plt.show(block = False)

# MFCC heatmap (12 vs 64)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12,5))
fig.suptitle(' ')

im = ax1.imshow(MFCC_12_pre_emphasized.T, origin='lower', aspect='auto', interpolation='nearest')
ax1.set_title('12 banks MFCC')
ax1.set_xlabel('frame')
ax1.set_ylabel('MFCC coefficient')

im = ax2.imshow(MFCC_64_pre_emphasized.T, origin='lower', aspect='auto', interpolation='nearest')
ax2.set_title('64 banks MFCC')
ax2.set_xlabel('frame')
ax2.set_ylabel('MFCC coefficient')

plt.tight_layout()
plt.show(block= False)


# inverse DCT (done for you)
inv_DCT_12_original = idct(MFCC_12_original, norm = 'ortho')
inv_DCT_64_original = idct(MFCC_64_original, norm = 'ortho')
inv_DCT_12_pre_emphasized = idct(MFCC_12_pre_emphasized, norm = 'ortho')
inv_DCT_64_pre_emphasized = idct(MFCC_64_pre_emphasized, norm = 'ortho')

# mag scale restoration:
###################
# YOUR CODE HERE

linear_scale_12_original = np.exp(inv_DCT_12_original.T)
linear_scale_64_original = np.exp(inv_DCT_64_original.T)
linear_scale_12_pre_emphasized = np.exp(inv_DCT_12_pre_emphasized.T)
linear_scale_64_pre_emphasized = np.exp(inv_DCT_64_pre_emphasized.T)
###################

# inverse convoluation against fbanks (mind the shapes of your matrices):
###################
# YOUR CODE HERE

inv_spectrogram_12_original = np.dot(pinv(fbanks_12), linear_scale_12_original)
inv_spectrogram_64_original = np.dot(pinv(fbanks_64), linear_scale_64_original)
inv_spectrogram_12_pre_emphasized = np.dot(pinv(fbanks_12), linear_scale_12_pre_emphasized)
inv_spectrogram_64_pre_emphasized = np.dot(pinv(fbanks_64), linear_scale_64_pre_emphasized)
    
###################

# signal restoration to time domain (You only have to finish griffinlim() in 'stft2audio_student.py'):
inv_audio_12_original = griffinlim(inv_spectrogram_12_original, n_iter=128, hop_length=frame_step, win_length=frame_length)
sf.write(os.path.join(BASE_DIR, 'reconstructed_12_original.wav'), inv_audio_12_original, samplerate=int(sr*512/frame_length))
reconstructed_spectrum_12_original = STFT(inv_audio_12_original, num_frames, num_FFT, frame_step, frame_length, len(inv_audio_12_original), verbose=False)

inv_audio_64_original = griffinlim(inv_spectrogram_64_original, n_iter=128, hop_length=frame_step, win_length=frame_length)
sf.write(os.path.join(BASE_DIR, 'reconstructed_64_original.wav'), inv_audio_64_original, samplerate=int(sr*512/frame_length))
reconstructed_spectrum_64_original = STFT(inv_audio_64_original, num_frames, num_FFT, frame_step, frame_length, len(inv_audio_64_original), verbose=False)

inv_audio_12_pre_emphasized = griffinlim(inv_spectrogram_12_pre_emphasized, n_iter=128, hop_length=frame_step, win_length=frame_length)
inv_audio_12_pre_emphasized = de_emphasis(inv_audio_12_pre_emphasized)
sf.write(os.path.join(BASE_DIR, 'reconstructed_12_pre_emphasized.wav'), inv_audio_12_pre_emphasized, samplerate=int(sr*512/frame_length))
reconstructed_spectrum_12_pre_emphasized = STFT(inv_audio_12_pre_emphasized, num_frames, num_FFT, frame_step, frame_length, len(inv_audio_12_pre_emphasized), verbose=False)

inv_audio_64_pre_emphasized = griffinlim(inv_spectrogram_64_pre_emphasized, n_iter=128, hop_length=frame_step, win_length=frame_length)
inv_audio_64_pre_emphasized = de_emphasis(inv_audio_64_pre_emphasized)
sf.write(os.path.join(BASE_DIR, 'reconstructed_64_pre_emphasized.wav'), inv_audio_64_pre_emphasized, samplerate=int(sr*512/frame_length))
reconstructed_spectrum_64_pre_emphasized = STFT(inv_audio_64_pre_emphasized, num_frames, num_FFT, frame_step, frame_length, len(inv_audio_64_pre_emphasized), verbose=False)

# scale and plot and compare original and reconstructed signals
# scale (done for you):
# TODO: Assign the spectrogram of the ORIGINAL signal to 'absolute_spectrum' for comparison.

absolute_spectrum_12_original = original_spectrum
absolute_spectrum_12_original = np.where(absolute_spectrum_12_original == 0, np.finfo(float).eps, absolute_spectrum_12_original)
absolute_spectrum_12_original = np.log(absolute_spectrum_12_original)
reconstructed_spectrum_12_original = np.where(reconstructed_spectrum_12_original == 0, np.finfo(float).eps, reconstructed_spectrum_12_original)
reconstructed_spectrum_12_original = np.log(reconstructed_spectrum_12_original)

absolute_spectrum_64_original = original_spectrum
absolute_spectrum_64_original = np.where(absolute_spectrum_64_original == 0, np.finfo(float).eps, absolute_spectrum_64_original)
absolute_spectrum_64_original = np.log(absolute_spectrum_64_original)
reconstructed_spectrum_64_original = np.where(reconstructed_spectrum_64_original == 0, np.finfo(float).eps, reconstructed_spectrum_64_original)
reconstructed_spectrum_64_original = np.log(reconstructed_spectrum_64_original)

absolute_spectrum_12_pre_emphasized = pre_emphasized_spectrum
absolute_spectrum_12_pre_emphasized = np.where(absolute_spectrum_12_pre_emphasized == 0, np.finfo(float).eps, absolute_spectrum_12_pre_emphasized)
absolute_spectrum_12_pre_emphasized = np.log(absolute_spectrum_12_pre_emphasized)
reconstructed_spectrum_12_pre_emphasized = np.where(reconstructed_spectrum_12_pre_emphasized == 0, np.finfo(float).eps, reconstructed_spectrum_12_pre_emphasized)
reconstructed_spectrum_12_pre_emphasized = np.log(reconstructed_spectrum_12_pre_emphasized)

absolute_spectrum_64_pre_emphasized = pre_emphasized_spectrum
absolute_spectrum_64_pre_emphasized = np.where(absolute_spectrum_64_pre_emphasized == 0, np.finfo(float).eps, absolute_spectrum_64_pre_emphasized)
absolute_spectrum_64_pre_emphasized = np.log(absolute_spectrum_64_pre_emphasized)
reconstructed_spectrum_64_pre_emphasized = np.where(reconstructed_spectrum_64_pre_emphasized == 0, np.finfo(float).eps, reconstructed_spectrum_64_pre_emphasized)
reconstructed_spectrum_64_pre_emphasized = np.log(reconstructed_spectrum_64_pre_emphasized)

#plot:
###################
# YOUR CODE HERE

# Plot the original and reconstructed spectrograms side by side
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle('Original vs. Reconstructed Spectrogram (Log Scale)')

ax1.imshow(np.log(original_spectrum + np.finfo(float).eps), origin='lower', aspect='auto', cmap='viridis')
ax1.set_title('Original Spectrogram')
ax1.set_xlabel('Frame')
ax1.set_ylabel('Frequency Bin')

ax2.imshow(reconstructed_spectrum_64_pre_emphasized, origin='lower', aspect='auto', cmap='viridis')
ax2.set_title('Reconstructed Spectrogram')
ax2.set_xlabel('Frame')
ax2.set_ylabel('Frequency Bin')

plt.tight_layout()
plt.show()

def SNR(source, signal):
    eps = np.finfo(float).eps
    signal_power = np.sum(source**2)
    noise_power = np.sum((source - signal)**2) + eps
    SNR = 10.0 * np.log10(signal_power / noise_power)
    
    return SNR

SNR_12_premphasized = SNR(original_spectrum,  np.exp(reconstructed_spectrum_12_pre_emphasized))
print(f"SNR of 12_pre = {SNR_12_premphasized}")

SNR_64_premphasized = SNR(original_spectrum,  np.exp(reconstructed_spectrum_64_pre_emphasized))
print(f"SNR of 64_pre = {SNR_64_premphasized}")

SNR_12_original = SNR(original_spectrum,  np.exp(reconstructed_spectrum_12_original))
print(f"SNR of 12_ori = {SNR_12_original}")

SNR_64_original = SNR(original_spectrum,  np.exp(reconstructed_spectrum_64_original))
print(f"SNR of 64_ori = {SNR_64_original}")


# ===================== DCT compression/decompression SNR experiment (follow user's pipeline) =====================
# This block strictly follows the existing pipeline in Lab2_student.py:
# inv-DCT -> exp -> pseudo-inverse of Mel banks -> Griffin-Lim -> STFT -> log/exp shaping -> SNR()
# We ONLY add dimensionality reduction by zeroing MFCCs beyond K. SNR() usage and comparison stay identical.

import numpy as np

def keep_first_k(mfcc_mat: np.ndarray, k: int) -> np.ndarray:
    """Zero out MFCC coefficients beyond the first k (frame-wise)."""
    k = max(0, min(k, mfcc_mat.shape[1]))
    out = np.zeros_like(mfcc_mat)
    out[:, :k] = mfcc_mat[:, :k]
    return out

def compress_decompress_and_measure(mfcc_mat, fbanks, is_pre_emphasized: bool, K: int, tag: str):
    """Run user's exact reconstruction chain and SNR computation after keeping only first K MFCCs."""
    # 1) inv-DCT (orthonormal) -> matches your existing code
    inv_dct = idct(mfcc_mat, norm='ortho')  # shape: (num_frames, num_filters)
    # 2) log->linear magnitude
    linear_scale = np.exp(inv_dct.T)
    # 3) invert Mel fbanks by pseudoinverse (same as your code)
    inv_spec = np.dot(pinv(fbanks), linear_scale)
    # 4) Griffin-Lim reconstruction
    inv_audio = griffinlim(inv_spec, n_iter=128, hop_length=frame_step, win_length=frame_length)
    # 5) De-emphasis only when the front-end had pre-emphasis (follow your pipeline)
    if is_pre_emphasized:
        inv_audio = de_emphasis(inv_audio)
    # 6) STFT of the reconstructed signal (same arguments as yours)
    recon_spec = STFT(inv_audio, num_frames, num_FFT, frame_step, frame_length, len(inv_audio), verbose=False)
    # 7) Prepare log-domain arrays exactly as you do before SNR()
    recon_spec = np.where(recon_spec == 0, np.finfo(float).eps, recon_spec)
    recon_log = np.log(recon_spec)
    # 8) Your SNR() compares original_spectrum to exp(reconstructed_log) — we do the same.
    snr_value = SNR(original_spectrum, np.exp(recon_log))
    print(f"[{tag}] K={K:>3d} -> SNR = {snr_value:.4f} dB")
    return snr_value

# K settings to try (you can change freely)
K_list_12 = [4, 6, 8, 10, 12]
K_list_64 = [8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60, 64]

print("\n=== DCT compression/decompression using user's SNR definition ===")

# Initialize dictionaries to store SNR values
snr_results = {
    '12banks_ori': {},
    '12banks_pre': {},
    '64banks_ori': {},
    '64banks_pre': {}
}

# ----- 12-filterbanks, ORIGINAL (no de-emphasis) -----
for K in K_list_12:
    mfcc_K = keep_first_k(MFCC_12_original, K)
    snr_value = compress_decompress_and_measure(mfcc_K, fbanks_12, is_pre_emphasized=False, K=K, tag="12banks_ori")
    snr_results['12banks_ori'][K] = snr_value

# ----- 12-filterbanks, PRE-EMPHASIZED (with de-emphasis) -----
for K in K_list_12:
    mfcc_K = keep_first_k(MFCC_12_pre_emphasized, K)
    snr_value = compress_decompress_and_measure(mfcc_K, fbanks_12, is_pre_emphasized=True, K=K, tag="12banks_pre")
    snr_results['12banks_pre'][K] = snr_value

# ----- 64-filterbanks, ORIGINAL -----
for K in K_list_64:
    mfcc_K = keep_first_k(MFCC_64_original, K)
    snr_value = compress_decompress_and_measure(mfcc_K, fbanks_64, is_pre_emphasized=False, K=K, tag="64banks_ori")
    snr_results['64banks_ori'][K] = snr_value

# ----- 64-filterbanks, PRE-EMPHASIZED -----
for K in K_list_64:
    mfcc_K = keep_first_k(MFCC_64_pre_emphasized, K)
    snr_value = compress_decompress_and_measure(mfcc_K, fbanks_64, is_pre_emphasized=True, K=K, tag="64banks_pre")
    snr_results['64banks_pre'][K] = snr_value

# Prepare data for plotting
K_12 = list(snr_results['12banks_ori'].keys())
snr_ori_12 = [snr_results['12banks_ori'][k] for k in K_12]
snr_pre_12 = [snr_results['12banks_pre'][k] for k in K_12]

K_64 = list(snr_results['64banks_ori'].keys())
snr_ori_64 = [snr_results['64banks_ori'][k] for k in K_64]
snr_pre_64 = [snr_results['64banks_pre'][k] for k in K_64]

# Plot SNR results
plt.figure(figsize=(12, 5))

# Left plot: 12 banks
plt.subplot(1, 2, 1)
plt.plot(K_12, snr_ori_12, marker='o', label='Original')
plt.plot(K_12, snr_pre_12, marker='s', label='Pre-emphasized')
plt.title('SNR vs K for 12 Banks')
plt.xlabel('Number of Coefficients')
plt.ylabel('SNR (dB)')
plt.legend()
plt.grid(True)

# Right plot: 64 banks
plt.subplot(1, 2, 2)
plt.plot(K_64, snr_ori_64, marker='o', label='Original')
plt.plot(K_64, snr_pre_64, marker='s', label='Pre-emphasized')
plt.title('SNR vs K for 64 Banks')
plt.xlabel('Number of Coefficients')
plt.ylabel('SNR (dB)')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()
