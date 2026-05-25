#%%
import os
import itertools
import librosa
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from glob import glob
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import KFold
from sklearn.svm import SVC, LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import HistGradientBoostingClassifier
from Lab3_111061220_functions import plot_confusion_matrix
from tqdm import tqdm

RANDSEED = 42 # setup random seed
CVFOLD = 5    # number of folds of cross validation
classNames = ['Dog bark', 'Rain', 'Sea waves', 'Baby cry',
              'Clock tick', 'Person sneeze', 'Helicopter', 'Chainsaw',
              'Rooster', 'Fire crackling']
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
#%%
##### Load data & Calculate features MFCC
labels = pd.read_csv(os.path.join(DATA_DIR, 'label.csv'))
nameToLabel = dict((row['filename'], row['label']) for idx, row in labels.iterrows())
trainFiles = sorted(glob(os.path.join(DATA_DIR, 'Train', '*', '*.ogg')))
testFiles = sorted(glob(os.path.join(DATA_DIR, 'Test', '*', '*.ogg')))
trainLabel = np.array([nameToLabel[os.path.basename(p)] for p in trainFiles])
testLabel = np.array([nameToLabel[os.path.basename(p)] for p in testFiles])

def feat_extraction(path):
    '''
    Input: path for a single file
    Output: 1D feature vector
    (1) Read file using librosa
    (2) Use librosa to calculate MFCC
    (3) Aggregate the 2D MFCC along time axis to 1D feature vector (ex: mean, std ...)
    '''
	######
	#CODE HERE

    y, sr = librosa.load(path, sr=None)
    
    # All the features in librosa.feature
    chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr)
    chroma_cqt = librosa.feature.chroma_cqt(y=y, sr=sr)
    chroma_cens = librosa.feature.chroma_cens(y=y, sr=sr)
    chroma_vqt =  librosa.feature.chroma_vqt(y=y, sr=sr, intervals=[1.0])
    melspectrogram = librosa.feature.melspectrogram(y=y, sr=sr)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
    rms = librosa.feature.rms(y=y)
    spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
    spectral_flatness = librosa.feature.spectral_flatness(y=y)
    spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    poly_features = librosa.feature.poly_features(y=y, sr=sr)
    tonnetz = librosa.feature.tonnetz(y=librosa.effects.harmonic(y), sr=sr)
    zero_crossing_rate = librosa.feature.zero_crossing_rate(y)
    
    tempo = np.abs(librosa.feature.tempo(y=y, sr=sr))
    tempogram = np.abs(librosa.feature.tempogram(y=y, sr=sr, win_length=157))
    fourier_tempogram = np.abs(librosa.feature.fourier_tempogram(y=y, sr=sr, win_length=157))
    tempogram_ratio = np.abs(librosa.feature.tempogram_ratio(y=y, sr=sr, win_length=157))

    stack_memory = librosa.feature.stack_memory(mfcc, n_steps=3)

    # Aggregate features
    chroma_stft_mean = np.mean(chroma_stft, axis=1)
    chroma_stft_std = np.std(chroma_stft, axis=1)
    chroma_cqt_mean = np.mean(chroma_cqt, axis=1)
    chroma_cqt_std = np.std(chroma_cqt, axis=1)
    chroma_cens_mean = np.mean(chroma_cens, axis=1)
    chroma_cens_std = np.std(chroma_cens, axis=1)
    chroma_vqt_mean = np.mean(chroma_vqt, axis=1)
    chroma_vqt_std = np.std(chroma_vqt, axis=1)
    melspectrogram_mean = np.mean(melspectrogram, axis=1)
    melspectrogram_std = np.std(melspectrogram, axis=1)
    mfcc_mean = np.mean(mfcc, axis=1)
    mfcc_std = np.std(mfcc, axis=1)
    rms_mean = np.mean(rms, axis=1)
    rms_std = np.std(rms, axis=1)
    spectral_centroid_mean = np.mean(spectral_centroid, axis=1)
    spectral_centroid_std = np.std(spectral_centroid, axis=1)
    spectral_bandwidth_mean = np.mean(spectral_bandwidth, axis=1)
    spectral_bandwidth_std = np.std(spectral_bandwidth, axis=1)
    spectral_contrast_mean = np.mean(spectral_contrast, axis=1)
    spectral_contrast_std = np.std(spectral_contrast, axis=1)
    spectral_flatness_mean = np.mean(spectral_flatness, axis=1)
    spectral_flatness_std = np.std(spectral_flatness, axis=1)
    spectral_rolloff_mean = np.mean(spectral_rolloff, axis=1)
    spectral_rolloff_std = np.std(spectral_rolloff, axis=1)
    poly_features_mean = np.mean(poly_features, axis=1)
    poly_features_std = np.std(poly_features, axis=1)
    tonnetz_mean = np.mean(tonnetz, axis=1)
    tonnetz_std = np.std(tonnetz, axis=1)
    zero_crossing_rate_mean = np.mean(zero_crossing_rate, axis=1)
    zero_crossing_rate_std = np.std(zero_crossing_rate, axis=1)
    
    tempo_mean = np.mean(tempo)
    temop_std = np.std(tempo)
    tempogram_mean = np.mean(tempogram, axis=1)
    tempogram_std = np.std(tempogram, axis=1)
    fourier_tempogram_mean = np.mean(fourier_tempogram, axis=1)
    fourier_tempogram_std = np.std(fourier_tempogram, axis=1)
    tempogram_ratio_mean = np.mean(tempogram_ratio, axis=1)
    tempogram_ratio_std = np.std(tempogram_ratio, axis=1)
    
    stack_memory_mean = np.mean(stack_memory, axis=1)
    stack_memory_std = np.std(stack_memory, axis=1)
    
    # Concatenate all features into a single feature vector
    features = np.hstack([chroma_stft_mean, chroma_stft_std,
                          chroma_cqt_mean, chroma_cqt_std,
                          chroma_cens_mean, chroma_cens_std,
                          chroma_vqt_mean, chroma_vqt_std,
                          melspectrogram_mean, melspectrogram_std,
                          mfcc_mean, mfcc_std,
                          rms_mean, rms_std,
                          spectral_centroid_mean, spectral_centroid_std,
                          spectral_bandwidth_mean, spectral_bandwidth_std,
                          spectral_contrast_mean, spectral_contrast_std,
                          spectral_flatness_mean, spectral_flatness_std,
                          spectral_rolloff_mean, spectral_rolloff_std,
                          poly_features_mean, poly_features_std,
                          tonnetz_mean, tonnetz_std,
                          zero_crossing_rate_mean, zero_crossing_rate_std,
                          tempo_mean, temop_std,
                          tempogram_mean, tempogram_std,
                          fourier_tempogram_mean, fourier_tempogram_std,
                          tempogram_ratio_mean, tempogram_ratio_std,
                          stack_memory_mean, stack_memory_std])
	######
    return features

def feat_extraction_basic(path):

    y, sr = librosa.load(path, sr=None)
    
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)

    mfcc_mean = np.mean(mfcc, axis=1)
    mfcc_std = np.std(mfcc, axis=1)
    
    features = np.hstack([mfcc_mean, mfcc_std])
    
    return features

# trainFeat = np.vstack([feat_extraction_basic(p) for p in trainFiles])
# testFeat = np.vstack([feat_extraction_basic(p) for p in testFiles])
trainFeat = np.vstack([feat_extraction(p) for p in tqdm(trainFiles, desc='Extract train features')])
testFeat = np.vstack([feat_extraction(p) for p in tqdm(testFiles, desc='Extract test features')])

#%%
##### Perform cross-validation
'''
(1) Use KFold to perform cross validation
(2) Normalize training set and testing set
(3) Collect result from each fold
(4) Calculate accuracy and confusion matrix
'''
X = trainFeat
y = trainLabel
Kf = KFold(n_splits=CVFOLD, shuffle=True, random_state=RANDSEED)
sc = StandardScaler()

y_dev_cv = []
y_predict_cv = []

for cvIdx, (trainIdx, devIdx) in enumerate(tqdm(list(Kf.split(range(len(X)))), desc='CV folds')):
	######
	#CODE HERE
 
    X_train, X_dev = X[trainIdx], X[devIdx]
    y_train, y_dev = y[trainIdx], y[devIdx]

    X_train_s = sc.fit_transform(X_train)
    X_dev_s = sc.transform(X_dev)

    # clf = SVC(kernel="linear", random_state=RANDSEED)
    clf = HistGradientBoostingClassifier(random_state=RANDSEED, l2_regularization=0.1,
                                         max_iter=256, warm_start=True,)
    clf.fit(X_train_s, y_train)

    y_pred = clf.predict(X_dev_s)

    y_dev_cv.extend(y_dev.tolist())
    y_predict_cv.extend(y_pred.tolist())
	######
    
accuracy = accuracy_score(y_dev_cv, y_predict_cv)
cm = confusion_matrix(y_dev_cv, y_predict_cv)

plot_confusion_matrix(cm , classNames)
print('ACC = ',  accuracy)
#%%
##### Predict on test set
'''
(1) Train a model based on your best parameters
(2) Prediction on test set
(3) Calculate accuracy and confusion matrix
'''
X_test = np.vstack(testFeat)
y_test = np.array(testLabel)

######
#CODE HERE

sc_full = StandardScaler()
X_train_full_s = sc_full.fit_transform(X)
# clf_full = SVC(kernel="linear", random_state=RANDSEED)
clf_full = HistGradientBoostingClassifier(random_state=RANDSEED, l2_regularization=0.1,
                                          max_iter=256, warm_start=True,)
                                          
clf_full.fit(X_train_full_s, y)

X_test_s = sc_full.transform(X_test)
y_pred_test = clf_full.predict(X_test_s)
######

accuracy = accuracy_score(y_test, y_pred_test)
cm = confusion_matrix(y_test, y_pred_test)

plot_confusion_matrix(cm , classNames)
print('ACC = ',  accuracy)

# %%
