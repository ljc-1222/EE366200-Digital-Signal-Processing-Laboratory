#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Lab4 Functions Module
Contains all functions for baby sound classification

@author: wschien
"""

import warnings
warnings.filterwarnings('ignore')

import os
import hashlib
import librosa
import itertools
import numpy as np
import pandas as pd
import multiprocessing
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor

from tqdm import tqdm
from sklearn.model_selection import KFold
from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score

# Constants
RANDSEED = 42
classNames = ['Crying', 'Laughing', 'Canonical', 'Non-canonical', 'Junk']


def MFCC_feat(y, sr):
    '''
    Extract MFCC features with multiple FFT sizes for robust audio classification.
    '''
    
    n_ffts = [128, 256, 512, 1024]
    
    feats = []
    
    for n_fft in n_ffts:
        
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=24, n_fft=n_fft)
        mfcc_mean = np.mean(mfcc, axis=1)
        mfcc_var = np.var(mfcc, axis=1)
        
        rms = librosa.feature.rms(y=y)
        rms_mean = np.mean(rms, axis=1)
        rms_var = np.var(rms, axis=1)
        
        chroma_cqt = librosa.feature.chroma_cqt(y=y, sr=sr)
        chroma_cqt_mean = np.mean(chroma_cqt, axis=1)   
        chroma_cqt_var = np.var(chroma_cqt, axis=1)
        
        spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr, n_fft=n_fft)
        spectral_contrast_mean = np.mean(spectral_contrast, axis=1)
        spectral_contrast_var = np.var(spectral_contrast, axis=1)
        
        spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr, n_fft=n_fft)
        spectral_rolloff_mean = np.mean(spectral_rolloff, axis=1)
        spectral_rolloff_var = np.var(spectral_rolloff, axis=1)
        
        melspectrogram = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=n_fft)
        mel_mean = np.mean(melspectrogram, axis=1)
        mel_var = np.var(melspectrogram, axis=1)
        
        tempo = np.abs(librosa.feature.tempo(y=y, sr=sr))
        tempo_mean = np.atleast_1d(np.mean(tempo))
        tempo_var = np.atleast_1d(np.var(tempo))
    
        feats.append(np.hstack((mfcc_mean, mfcc_var,
                                rms_mean, rms_var,
                                chroma_cqt_mean, chroma_cqt_var,
                                spectral_contrast_mean, spectral_contrast_var,
                                spectral_rolloff_mean, spectral_rolloff_var,
                                mel_mean, mel_var,
                                tempo_mean, tempo_var
                            )))
        
    return np.hstack(feats)


def data_augmentation(file):
    '''
    Apply data augmentation to audio files while preserving essential features.
    '''
    
    y, sr = librosa.load(file, sr=None)
    original_length = len(y)
    duration = len(y) / sr
    
    # Randomly select augmentation type
    aug_type = np.random.randint(0, 4)
    
    if aug_type == 0:
        # Volume scaling: 0.7x to 1.3x
        volume_factor = 0.7 + 0.6 * np.random.rand()
        y = y * volume_factor
        
    elif aug_type == 1:
        # Time stretching: 0.9x to 1.1x
        stretch_rate = 0.9 + 0.2 * np.random.rand()
        y = librosa.effects.time_stretch(y, rate=stretch_rate)
        
    elif aug_type == 2:
        # Pitch shift: -0.2 to +0.2 semitones
        pitch_steps = -0.2 + 0.4 * np.random.rand()
        y = librosa.effects.pitch_shift(y, sr=sr, n_steps=pitch_steps)
        
    else:
        # Additive noise: 1-3% noise level
        noise_level = 0.01 + 0.02 * np.random.rand()
        noise = np.random.normal(0, noise_level, len(y))
        y = y + noise
    
    # Maintain original length
    if len(y) != original_length:
        if len(y) > original_length:
            y = y[:original_length]
        else:
            y = np.pad(y, (0, original_length - len(y)), mode='constant')
    
    # Normalize to prevent clipping
    if np.max(np.abs(y)) > 0:
        y = y / np.max(np.abs(y)) * 0.95
    
    return MFCC_feat(y, sr)


def get_original_cache_path(file_path, cache_dir='./feature_cache'):
    """Generate cache path for original features."""
    os.makedirs(cache_dir, exist_ok=True)
    hash_input = f"{file_path}_original"
    file_hash = hashlib.md5(hash_input.encode()).hexdigest()[:12]
    filename = os.path.basename(file_path).replace('.wav', f'_original_{file_hash}.npy')
    return os.path.join(cache_dir, filename)


def get_augmented_cache_path(file_path, aug_idx, cache_dir='./feature_cache'):
    """Generate cache path for augmented features."""
    os.makedirs(cache_dir, exist_ok=True)
    hash_input = f"{file_path}_aug{aug_idx}"
    file_hash = hashlib.md5(hash_input.encode()).hexdigest()[:12]
    filename = os.path.basename(file_path).replace('.wav', f'_aug{aug_idx}_{file_hash}.npy')
    return os.path.join(cache_dir, filename)


def cache_original_features(file_path, label, cache_dir='./feature_cache'):
    """Load or extract and cache original features."""
    cache_path = get_original_cache_path(file_path, cache_dir)
    
    if os.path.exists(cache_path):
        return np.load(cache_path)
    
    original_feat = MFCC_feat(*librosa.load(file_path, sr=None))
    np.save(cache_path, original_feat)
    return original_feat


def cache_augmented_features(file_path, aug_idx, cache_dir='./feature_cache'):
    """Load or extract and cache augmented features."""
    cache_path = get_augmented_cache_path(file_path, aug_idx, cache_dir)
    
    if os.path.exists(cache_path):
        return np.load(cache_path)
    
    augmented_feat = data_augmentation(file_path)
    np.save(cache_path, augmented_feat)
    return augmented_feat


def process_single_file_parallel(args):
    """
    Process a single file and extract features for parallel execution.
    
    Args:
        args: tuple of (file_path, label, cache_dir, num_augmentations)
        
    Returns:
        tuple: (features_list, labels_list)
    """
    file_path, label, cache_dir, num_augmentations = args
    features_list = []
    labels_list = []
    
    try:
        # Extract original features
        original_feat = cache_original_features(file_path, label, cache_dir)
        features_list.append(original_feat)
        labels_list.append(label)
        
        # Extract augmented features
        for aug_idx in range(num_augmentations):
            augmented_feat = cache_augmented_features(file_path, aug_idx, cache_dir)
            features_list.append(augmented_feat)
            labels_list.append(label)
            
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return [], []
    
    return features_list, labels_list


def load_cached_features_parallel(file_paths, labels, cache_dir='./feature_cache', num_augmentations=12, n_jobs=None):
    """
    Load cached features using parallel processing.
    
    Args:
        file_paths (list): List of file paths
        labels (list): List of corresponding labels
        cache_dir (str): Directory to store cached features
        num_augmentations (int): Number of augmented samples per file
        n_jobs (int): Number of parallel jobs (None for auto-detect)
        
    Returns:
        tuple: (X, y) where X is the feature matrix and y is the label array
    """
    if n_jobs is None:
        n_jobs = min(multiprocessing.cpu_count(), len(file_paths))
    
    print(f"Using {n_jobs} parallel workers for feature extraction...")
    
    # Prepare arguments for parallel processing
    args = [(file_path, label, cache_dir, num_augmentations) 
            for file_path, label in zip(file_paths, labels)]
    
    all_features = []
    all_labels = []
    
    # Process files in parallel
    with ProcessPoolExecutor(max_workers=n_jobs) as executor:
        results = list(tqdm(executor.map(process_single_file_parallel, args), 
                          total=len(args), 
                          desc='Loading cached features in parallel',
                          leave=False,
                          dynamic_ncols=True))
    
    # Collect successful results
    for features_list, labels_list in results:
        if features_list:
            all_features.extend(features_list)
            all_labels.extend(labels_list)
    
    if not all_features:
        raise ValueError("No features were successfully extracted!")
    
    return np.vstack(all_features), np.array(all_labels)


def load_cached_features_efficient(file_paths, labels, cache_dir='./feature_cache', num_augmentations=12):
    """
    Load cached features using parallel processing.
    
    Args:
        file_paths (list): List of file paths
        labels (list): List of corresponding labels
        cache_dir (str): Directory to store cached features
        num_augmentations (int): Number of augmented samples per file
        
    Returns:
        tuple: (X, y) where X is the feature matrix and y is the label array
    """
    return load_cached_features_parallel(file_paths, labels, cache_dir, num_augmentations)


def load_cached_features(file_paths, labels, cache_dir='./feature_cache', num_augmentations=12):
    """
    Load cached features for multiple files.
    
    Args:
        file_paths (list): List of file paths
        labels (list): List of corresponding labels
        cache_dir (str): Directory to store cached features
        num_augmentations (int): Number of augmented samples per file
        
    Returns:
        tuple: (X, y) where X is the feature matrix and y is the label array
    """
    return load_cached_features_efficient(file_paths, labels, cache_dir, num_augmentations)


def extract_validation_features(file_path, label, num_augmentations=8):
    """
    Extract features for validation: original + augmented samples.
    
    Args:
        file_path (str): Path to the audio file
        label (int): Label for the file
        num_augmentations (int): Number of augmented samples to generate
        
    Returns:
        tuple: (features_list, labels_list) containing original and augmented features
    """
    features_list = []
    labels_list = []
    
    # Extract original features
    features_list.append(MFCC_feat(*librosa.load(file_path, sr=None)))
    labels_list.append(label)
    
    # Extract augmented features
    for _ in range(num_augmentations):
        features_list.append(data_augmentation(file_path))
        labels_list.append(label)
    
    return features_list, labels_list


def pre_extract_test_features(test_path, cache_dir='./feature_cache', num_augmentations=8):
    """
    Pre-extract and cache all features for the test set.
    
    Args:
        test_path (list): List of test file paths
        cache_dir (str): Directory to store cached features
        num_augmentations (int): Number of augmented samples per file
        
    Returns:
        tuple: (X, y) where X is the feature matrix and y is the label array
    """
    print("Pre-extracting test features and caching them...")
    print("This will take some time on the first run, but subsequent runs will be much faster.")
    
    # Load cached features (will extract if not cached)
    dummy_labels = [0] * len(test_path)
    X, y = load_cached_features(test_path, dummy_labels, cache_dir, num_augmentations)
    
    print(f"Test feature extraction completed!")
    print(f"Total test features shape: {X.shape}")
    print(f"Total test labels shape: {y.shape}")
    print(f"Test features cached in: {cache_dir}")
    
    return X, y


def cross_val(cv, train_path, train_label):
    '''
    Cross validation using fixed settings: RobustScaler(10-90), 300 iter, 127 bins, 0.1 lr, 31 leaf, 12 aug, 5 ens.
    
    Dataset distribution:
    Crying: 243, Laughing: 46, Canonical: 444, Non-canonical: 1437, Junk: 1826 (Total: 3996)
    
    Process:
    1. Split data into cv folds
    2. For each fold: train on 4 folds, validate on 1 fold
    3. Use original + 12 augmented samples per file for training (13 total)
    4. Use original + 8 augmented samples per file for validation (9 total)
    5. Train 5 ensemble models with different random seeds
    6. Vote predictions from 9 samples using 5 models
    7. Evaluate performance on validation set
    
    Args:
        cv (int): Number of cross-validation folds
        train_path (list): List of training file paths
        train_label (list): List of training labels
        return_detailed_results (bool): If True, return detailed results per fold
        
    Returns:
        tuple: (acc, f1, cm) or (acc, f1, cm, detailed_results) if return_detailed_results=True
    '''
    
    Kf = KFold(n_splits=cv, shuffle=True, random_state=RANDSEED)
    sc = RobustScaler(quantile_range=(10, 90))
    y_dev_cv = []
    y_predict_cv = []
    
    for fold_idx, (train_index, dev_index) in enumerate(tqdm(Kf.split(train_path), desc='Cross validation', leave=False, dynamic_ncols=True, total=cv)):
        trainFiles = [train_path[i] for i in train_index]
        trainLabel = [train_label[i] for i in train_index]
        devFiles = [train_path[i] for i in dev_index]
        devLabel = [train_label[i] for i in dev_index]
        
        print(f"\nFold {fold_idx + 1}/{cv}: Training on {len(trainFiles)} files, validating on {len(devFiles)} files")
        
        # Extract training features (original + 12 augmented per file)
        print("Extracting training features (original + 8 augmented per file)...")
        trainFeat, trainLabel = load_cached_features(trainFiles, trainLabel, num_augmentations=8)
        
        # Train ensemble classifiers
        print("Training ensemble classifiers (5 models with different random seeds)...")
        np.random.seed(RANDSEED)
        sc.fit(trainFeat)
        X_train_scaled = sc.transform(trainFeat)
        
        # Train 5 HGBC models with different random seeds
        clf_models = []
        for i in range(5):
            random_seed = ((RANDSEED * (i + 1)) ** 2) % 4294967296
            clf_model = HistGradientBoostingClassifier(
                random_state=random_seed,
                warm_start=True,
                max_iter=300,
                early_stopping=True,
                max_bins=127,
                learning_rate=0.1,
                max_leaf_nodes=31,
                verbose=1
            )
            clf_model.fit(X_train_scaled, trainLabel)
            clf_models.append(clf_model)
            print(f"  Trained model {i+1}/5 with random_seed={random_seed}")
        
        # Validate on dev files using cached features (same as testing phase)
        print("Pre-extracting validation features (original + 8 augmented per file)...")
        X_dev_all, _ = pre_extract_test_features(devFiles, num_augmentations=8)
        
        print("Validating on dev files using cached features...")
        fold_predictions = []
        fold_true_labels = []
        samples_per_file = 8 + 1  # 8 augmented + 1 original for inference
        
        for i in tqdm(range(0, len(X_dev_all), samples_per_file), 
                      desc=f'Fold {fold_idx + 1} validation', leave=False, dynamic_ncols=True, total=len(devFiles)):
            # Get samples for current file
            file_features = X_dev_all[i:i+samples_per_file]
            X_test_scaled = sc.transform(file_features)
            
            # Ensemble prediction: 5 models vote on 9 samples each
            model_predictions = []
            
            for model_idx, clf_model in enumerate(clf_models):
                try:
                    # Each model votes on 9 samples (1 original + 8 augmented)
                    model_probs = clf_model.predict_proba(X_test_scaled)
                    log_probs = np.log(model_probs + 1e-12)
                    model_scores = np.sum(log_probs, axis=0)
                    model_pred = int(np.argmax(model_scores))
                    model_predictions.append(model_pred)
                except Exception:
                    # Fallback to majority vote
                    model_preds = clf_model.predict(X_test_scaled)
                    model_pred = int(np.bincount(model_preds).argmax())
                    model_predictions.append(model_pred)
            
            # Vote among the 5 models
            pred_label = int(np.bincount(model_predictions).argmax())
            
            # Store results
            fold_predictions.append(pred_label)
            fold_true_labels.append(devLabel[i // samples_per_file])
        
        # Calculate fold metrics
        fold_acc = accuracy_score(fold_true_labels, fold_predictions)
        fold_f1 = f1_score(fold_true_labels, fold_predictions, average='macro')
        fold_cm = confusion_matrix(fold_true_labels, fold_predictions, labels=range(len(classNames)))
        
        print(f"Fold {fold_idx + 1} Results - Accuracy: {fold_acc:.4f}, F1: {fold_f1:.4f}")
        
        # Append to overall results
        y_dev_cv.extend(fold_true_labels)
        y_predict_cv.extend(fold_predictions)

    # Calculate overall metrics
    acc = accuracy_score(y_dev_cv, y_predict_cv)
    f1 = f1_score(y_dev_cv, y_predict_cv, average='macro')
    cm = confusion_matrix(y_dev_cv, y_predict_cv, labels=range(len(classNames)))
    
    # Plot confusion matrix
    print("\nPlotting Cross-Validation Confusion Matrix...")
    plot_confusion_matrix(cm, classNames, 
                         title='Cross-Validation Confusion Matrix',
                         normalize=False)
    
    return acc, f1, cm


def train_and_predict_single_model(train_path, train_label, test_path):
    """
    Train ensemble model and generate predictions using fixed settings.
    
    Args:
        train_path (list): List of training file paths
        train_label (list): List of training labels
        test_path (list): List of test file paths
        
    Returns:
        str: Path to the generated CSV file
    """
    print(f"\n{'='*80}")
    print(f"Output file: result.csv")
    print(f"{'='*80}")
    
    # Fixed parameters
    sc = RobustScaler(quantile_range=(10, 90))
    num_augmentations = 8
    ensemble_size = 5
    
    # Extract training features
    print(f"Extracting training features (original + {num_augmentations} augmented per file)...")
    X_train, y_train = load_cached_features(train_path, train_label, num_augmentations=num_augmentations)
    print(f"Training features shape: {X_train.shape}")
    print(f"Training labels shape: {y_train.shape}")
    
    # Scale training data
    sc.fit(X_train)
    X_train_scaled = sc.transform(X_train)
    
    # Train ensemble classifiers
    print(f"Training ensemble classifiers ({ensemble_size} models with different random seeds)...")
    clf_models = []
    
    for i in range(ensemble_size):
        random_seed = ((RANDSEED * (i + 1)) ** 2) % 4294967296
        clf_model = HistGradientBoostingClassifier(
            random_state=random_seed,
            warm_start=True,
            max_iter=300,
            early_stopping=True,
            max_bins=127,
            learning_rate=0.1,
            max_leaf_nodes=31,
            verbose=1
        ) 
        clf_model.fit(X_train_scaled, y_train)
        clf_models.append(clf_model)
        print(f"  Trained model {i+1}/{ensemble_size}")
    
    print("Ensemble training completed!")
    
    # Extract test features
    print(f"Pre-extracting test features (original + 8 augmented per file)...")
    X_test_all, _ = pre_extract_test_features(test_path, num_augmentations = 8)
    
    # Predict on test set
    print("Predicting on test set...")
    y_pred = []
    samples_per_file = 8 + 1  # 8 augmented + 1 original for inference
    
    for i in tqdm(range(0, len(X_test_all), samples_per_file), 
                  desc='Predict test files', leave=False, dynamic_ncols=True, total=len(test_path)):
        # Get samples for current file
        file_features = X_test_all[i:i+samples_per_file]
        X_test_scaled = sc.transform(file_features)
        
        # Ensemble prediction
        model_predictions = []
        
        for clf_model in clf_models:
            try:
                model_probs = clf_model.predict_proba(X_test_scaled)
                log_probs = np.log(model_probs + 1e-12)
                model_scores = np.sum(log_probs, axis=0)
                model_pred = int(np.argmax(model_scores))
                model_predictions.append(model_pred)
            except Exception:
                model_preds = clf_model.predict(X_test_scaled)
                model_pred = int(np.bincount(model_preds).argmax())
                model_predictions.append(model_pred)
        
        # Vote among models
        pred_label = int(np.bincount(model_predictions).argmax())
        y_pred.append(pred_label)
    
    # Save results to CSV
    output_filename = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'result.csv')
    results = pd.DataFrame({
        'file_name': [os.path.basename(f) for f in test_path],
        'prediction': [classNames[p] for p in y_pred]
    })
    
    results.to_csv(output_filename, index=False)
    print(f'Saved predictions to {output_filename}')
    
    # Print prediction summary
    prediction_counts = pd.Series([classNames[p] for p in y_pred]).value_counts()
    print("Prediction distribution:")
    for class_name, count in prediction_counts.items():
        print(f"  {class_name}: {count} files ({count/len(y_pred)*100:.1f}%)")
    
    return output_filename

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix')

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=90)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()
