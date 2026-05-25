import numpy as np
from librosa import stft, istft

def griffinlim(S, n_iter=64, hop_length=None, win_length=None, window="hamming", center=True, dtype=None,
    length=None, pad_mode="reflect", momentum=0.99):

    # Infer n_fft from the spectrogram shape
    n_fft = 2 * (S.shape[0] - 1)

    # using complex64 will keep the result to minimal necessary precision
    angles = np.empty(S.shape, dtype=np.complex64)

    # randomly initialize the phase (simply use j as imaginary unit. e.g. A = 2j + 2):
    ###################
    # YOUR CODE HERE
    
    random_phases=np.random.uniform(0, 2 * np.pi, S.shape)
    angles = np.exp(1j* random_phases)
    ###################

    # And initialize the previous iterate to 0
    rebuilt = 0.0

    for _ in range(n_iter):
        # Store the previous iterate
        tprev = rebuilt

        # Invert with our current estimate of the phases
        inverse = istft(S * angles, hop_length=hop_length, win_length=win_length, window=window,
            center=center, dtype=dtype, length=length,)

        # Rebuild the spectrogram
        rebuilt = stft(inverse, n_fft=n_fft, hop_length=hop_length, win_length=win_length, window=window,
            center=center,pad_mode=pad_mode,)

        # Update our phase estimates
        # Momentum must be between 0 and 1 (0.99 is advised)
        angles[:] = rebuilt - (momentum / (1 + momentum)) * tprev
        angles[:] /= np.abs(angles) + 1e-16

    # Return the final phase estimates:
    ###################
    # YOUR CODE HERE
    
    final_inverse = istft(S * angles, hop_length=hop_length, win_length=win_length, window=window, 
        center=center, dtype=dtype, length=length)
    
    return final_inverse
    ###################
    
def pghi(S, hop_length, win_length, window="hamming", center=True, dtype=None, length=None, pad_mode="reflect"):
    n_fft = 2 * (S.shape[0] - 1)
    eps = 1e-8

    log_S = np.log(S + eps)
    dlogS_dt = np.gradient(log_S, axis=1)
    dlogS_df = np.gradient(log_S, axis=0)

    omega = 2 * np.pi * np.arange(S.shape[0])[:, None] * hop_length / n_fft
    phase_time = -dlogS_df * (n_fft / hop_length)
    phase_freq = dlogS_dt * (n_fft / win_length)

    phase = np.cumsum(phase_time, axis=1) + np.cumsum(phase_freq, axis=0)
    phase = np.mod(phase, 2*np.pi)
    phase = np.nan_to_num(phase)

    complex_spec = S * np.exp(1j * phase)
    y = istft(complex_spec, hop_length=hop_length, win_length=win_length,
              window=window, center=center, dtype=dtype, length=length)
    return y

def griffinlim_with_pghi_init(S, n_iter=128, hop_length=None, win_length=None,
                              window="hamming", center=True, dtype=None,
                              length=None, pad_mode="reflect", momentum=0.99):
    """
    Griffin-Lim with PGHI initialization.
    """

    # 1) PGHI initialization
    y0 = pghi(S, hop_length, win_length, window=window,
              center=center, dtype=dtype, length=length, pad_mode=pad_mode)

    # 2) STFT of initial signal
    n_fft = 2 * (S.shape[0] - 1)
    rebuilt0 = stft(y0, n_fft=n_fft, hop_length=hop_length, win_length=win_length,
                    window=window, center=center, pad_mode=pad_mode)
    angles = rebuilt0 / (np.abs(rebuilt0) + 1e-16)

    # 3) Run standard GLA iterations with PGHI init
    rebuilt = 0.0
    for _ in range(n_iter):
        tprev = rebuilt
        inverse = istft(S * angles, hop_length=hop_length, win_length=win_length,
                        window=window, center=center, dtype=dtype, length=length)
        rebuilt = stft(inverse, n_fft=n_fft, hop_length=hop_length, win_length=win_length,
                       window=window, center=center, pad_mode=pad_mode)
        angles[:] = rebuilt - (momentum / (1 + momentum)) * tprev
        angles[:] /= np.abs(angles) + 1e-16

    final_inverse = istft(S * angles, hop_length=hop_length, win_length=win_length,
                          window=window, center=center, dtype=dtype, length=length)
    return final_inverse
