'''
@Modified by Paul Cho; 10th, Nov, 2020

For NTHU DSP Lab 2025 Autumn
'''

import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt
import os
from librosa.filters import mel as librosa_mel_fn
from scipy.fftpack import dct

from Lab1_functions_student import pre_emphasis, STFT, mel2hz, hz2mel, get_filter_banks

filename = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'audio.wav')
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

##########################
'''
Part I:
(1) Perform STFT on the source signal to obtain one spectrogram (with the provided STFT() function)
(2) Pre-emphasize the source signal with pre_emphasis()
(3) Perform STFT on the pre-emphasized signal to obtain the second spectrogram
(4) Plot the two spectrograms together to observe the effect of pre-emphasis

hint for plotting:
you can use "plt.subplots()" to plot multiple figures in one.
you can use "axis.pcolor" of matplotlib in visualizing a spectrogram. 
'''
#YOUR CODE STARTS HERE:

# step(1): Perform STFT on the source signal to obtain one spectrogram
# STFT converts overlapping time-domain frames into frequency-domain magnitude (and possibly phase) representations.
# Expected output matrix shape: (frequency_bins, num_frames). Each column corresponds to one analysis frame.
# Parameters:
#   time_signal: raw waveform
#   num_frames / frame_length / frame_step: framing control
#   num_FFT: FFT size determining frequency resolution
#   signal_length: original signal length for boundary handling
#   verbose: show internal info for debugging/learning
first_spectrum = STFT(time_signal   = source_signal,
                      num_frames    = num_frames,
                      num_FFT       = num_FFT,
                      frame_step    = frame_step,
                      frame_length  = frame_length,
                      signal_length = signal_length,
                      verbose       = True)

# step(2): Pre-emphasize the source signal with pre_emphasis()
# Pre-emphasis is a first-order high-pass filter: y[n] = x[n] - a * x[n-1], emphasizing high-frequency components
# (compensates for the natural spectral tilt of voiced speech / many audio signals).
pre_emphasized_signal = pre_emphasis(signal = source_signal, coefficient = emphasis_coeff)

# step(3): Perform STFT on the pre-emphasized signal to obtain the second spectrogram
# Same STFT process; comparison with the original will highlight increased energy in higher frequency bins.
second_spectrum = STFT(time_signal  = pre_emphasized_signal,
                      num_frames    = num_frames,
                      num_FFT       = num_FFT,
                      frame_step    = frame_step,
                      frame_length  = frame_length,
                      signal_length = signal_length,
                      verbose       = True)

# step(4): Plot the two spectrograms together to observe the effect of pre-emphasis
# Using imshow: x-axis = frame index, y-axis = frequency bin index; origin='lower' so low freq at bottom.
# Note: No log scaling here, so dynamic range differences may be large.
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle('Original signal vs. Pre-emphasized signal')

img1 = ax1.imshow(first_spectrum, origin='lower', aspect='auto')
ax1.set_xlabel('frame')
ax1.set_ylabel('frequency band')
ax1.set_xlim(0, first_spectrum.shape[1])
ax1.set_ylim(0, first_spectrum.shape[0])

img2 = ax2.imshow(second_spectrum, origin='lower', aspect='auto')
ax2.set_xlabel('frame')
ax2.set_ylabel('frequency band')
ax2.set_xlim(0, second_spectrum.shape[1])
ax2.set_ylim(0, second_spectrum.shape[0])

plt.tight_layout()
plt.show(block= False)

#YOUR CODE ENDS HERE;
##########################

'''
Head to the import source 'Lab1_functions_student.py' to complete these functions:
mel2hz(), hz2mel(), get_filter_banks()
'''
# get Mel-scaled filter
fbanks = get_filter_banks(num_bands, num_FFT , sr, freq_min, freq_max)

##########################

'''
Part II:
(1) Convolve the pre-emphasized signal with the filter
(2) Convert magnitude to logarithmic scale
(3) Perform Discrete Cosine Transform (dct) as a process of information compression to obtain MFCC
    (already implemented for you, just notice this step is here and skip to the next step)
(4) Plot the filter banks alongside the MFCC
'''
#YOUR CODE STARTS HERE:

# step(1): Convolve the pre-emphasized signal with the filter
# We already have the frame-wise spectrum (second_spectrum). Applying Mel filter banks is a weighted summation across
# frequency bins: (num_bands x freq_bins) dot (freq_bins x num_frames) -> (num_bands x num_frames) energy matrix.
# This simulates critical-band integration approximating human auditory frequency selectivity.
filter_energies = np.dot(fbanks, second_spectrum)

# step(2): Convert magnitude to logarithmic scale
# Log compression approximates human loudness perception and stabilizes variance for later DCT.
# NOTE: If zeros may appear, typically a small epsilon (e.g., 1e-10) is added to avoid -inf. Here we assume >0.
features = np.log(filter_energies.T)

# step(3): Discrete Cosine Transform
# DCT decorrelates log filter-bank energies; MFCC keeps only the lower cepstral coefficients (most info for timbre).
# Taking the first num_bands coefficients (could also choose fewer for dimensionality reduction).
MFCC = dct(features, norm = 'ortho')[:,:num_bands]
# equivalent to Matlab dct(x)
# The numpy array [:,:] stands for everything from the beginning to end.

#########################

# Select one time frame randomly to visualize individual cepstral coefficient distribution.
rand_idx = np.random.randint(0, MFCC.shape[0])  
mfcc_frame = MFCC[rand_idx]                   

plt.figure(figsize=(6,4))
plt.plot(np.arange(len(mfcc_frame)), mfcc_frame)
plt.title('MFCC of a random frame')
plt.xlabel('Cepstral Coefficient')
plt.ylabel('Magnitude')
plt.tight_layout()
plt.show(block = False)

##########################

# step(4): Plot the filter banks alongside the MFCC
# Left: Each Mel filter's triangular weighting over linear frequency (converted to kHz for readability).
# Right: MFCC time-frequency (actually coefficient vs frame) heatmap. Transposed so coefficient index is on y-axis.
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12,5))
fig.suptitle('Mel-scaled filter banks and MFCC.')

freqs_khz = np.linspace(0, sr / 2, fbanks.shape[1]) / 1000.0  # Linear frequency axis (Nyquist) -> kHz for plotting

for i in range(fbanks.shape[0]):
    ax1.plot(freqs_khz, fbanks[i])
ax1.set_xlim(freqs_khz[0], freqs_khz[-1])

ax1.set_xlabel('frequency (kHz)')
ax1.set_ylabel('Mel-scale filter banks')

im = ax2.imshow(MFCC.T, origin='lower', aspect='auto', interpolation='nearest')
ax2.set_xlabel('frame')
ax2.set_ylabel('MFCC coefficient')

plt.tight_layout()
plt.show()

#YOUR CODE ENDS HERE;
##########################
