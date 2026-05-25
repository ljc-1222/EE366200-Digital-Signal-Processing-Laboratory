import numpy as np

def pre_emphasis(signal, coefficient = 0.95):

    return np.append(signal[0], signal[1:] - coefficient*signal[:-1])

def de_emphasis(signal, coefficient = 0.95):
    de_emphasized_signal = np.zeros(signal.shape)
    de_emphasized_signal[0] = signal[0]
    for n in range(1, len(signal)):
        de_emphasized_signal[n] = coefficient * de_emphasized_signal[n-1] + signal[n]

    return de_emphasized_signal

def STFT(time_signal, num_frames, num_FFT, frame_step, frame_length, signal_length, verbose=False):
    padding_length = int((num_frames - 1) * frame_step + frame_length)
    padding_zeros = np.zeros((padding_length - signal_length,))
    padded_signal = np.concatenate((time_signal, padding_zeros))

    # split into frames
    indices = np.tile(np.arange(0, frame_length), (num_frames, 1)) + np.tile(np.arange(0, num_frames*frame_step, frame_step), (frame_length, 1)).T
    indices = np.array(indices,dtype=np.int32)

    # slice signal into frames
    frames = padded_signal[indices]
    # apply window to the signal
    frames *= np.hamming(frame_length)

    # FFT
    complex_spectrum = np.fft.rfft(frames, num_FFT).T
    print(complex_spectrum.shape)
    absolute_spectrum = np.abs(complex_spectrum)
    
    if verbose:
        print('Signal length :{} samples.'.format(signal_length))
        print('Frame length: {} samples.'.format(frame_length))
        print('Frame step  : {} samples.'.format(frame_step))
        print('Number of frames: {}.'.format(len(frames)))
        print('Shape after FFT: {}.'.format(absolute_spectrum.shape))

    return absolute_spectrum


def mel2hz(mel):
    '''
    Transfer Mel scale to Hz scale
    '''
    ###################
    # YOUR CODE HERE
    
    # Using the formula to transform
    hz = 700 * (10 ** (mel / 2595) - 1)
    
    ###################
    
    return hz

def hz2mel(hz):
    '''
    Transfer Hz scale to Mel scale
    '''
    ###################
    # YOUR CODE HERE
    
    # Using the formula to transform
    mel = 2595 * np.log10(1 + hz / 700)
    
    ###################
    
    return mel

def get_filter_banks(num_filters, num_FFT, sample_rate, freq_min = 0, freq_max = None):
    ''' Mel Bank
    num_filters: filter numbers
    num_FFT: number of FFT quantization values
    sample_rate: as the name suggests
    freq_min: the lowest frequency that mel frequency include
    freq_max: the Highest frequency that mel frequency include
    '''
    # convert from hz scale to mel scale
    low_mel = hz2mel(freq_min)
    high_mel = hz2mel(freq_max)

    # define freq-axis
    mel_freq_axis = np.linspace(low_mel, high_mel, num_filters + 2)
    hz_freq_axis = mel2hz(mel_freq_axis)

    # Mel triangle bank design (Triangular band-pass filter banks)
    bins = np.floor((num_FFT + 1) * hz_freq_axis / sample_rate)
    fbanks = np.zeros((num_filters, int(num_FFT / 2 + 1)))

    ###################
    # YOUR CODE HERE
    
    for m in range(1, num_filters + 1):
        
        # f_m_start: left bin index (start of this filter)
        # f_m: center bin index (peak of this filter)
        # f_m_end: right bin index (end of this filter)
        
        f_m_start = int(bins[m - 1]) 
        f_m = int(bins[m])            
        f_m_end = int(bins[m + 1])

        # Construct the rising edge of the triangular filter (from 0 to 1)
        for k in range(f_m_start, f_m):
            # Linear interpolation from 0 at f_m_start to 1 at f_m
            fbanks[m - 1, k] = (k - f_m_start) / float(f_m - f_m_start)
                
        # Construct the falling edge of the triangular filter (from 1 to 0)
        for k in range(f_m, f_m_end):
            # Linear interpolation from 1 at f_m down to 0 at f_m_end
            fbanks[m - 1, k] = (f_m_end - k) / float(f_m_end - f_m)
                
    ###################

    return fbanks
