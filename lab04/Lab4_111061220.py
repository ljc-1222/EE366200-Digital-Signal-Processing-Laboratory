#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Lab4 Main Script
Baby Sound Classification using ensemble learning

@author: wschien
"""

import warnings
warnings.filterwarnings('ignore')

import os
import pandas as pd
from glob import glob

# Import functions from the functions module
from Lab4_111061220_functions import (
    cross_val, 
    train_and_predict_single_model, 
    classNames
)

# Configuration
DataPath = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    '2025-dsp-lab-detecting-baby-sounds',
    'Baby_Data'
)
CVFOLD = 5  # number of folds of cross validation

# Load training and test data
train_path = sorted(glob(os.path.join(DataPath, 'wav_train', 'train*.wav')))
test_path = sorted(glob(os.path.join(DataPath, 'wav_dev', 'dev*.wav')))

# Load labels
labels = pd.read_csv(os.path.join(DataPath, 'label_raw_train.csv'))
name2label = dict((row['file_name'], row['label']) for idx, row in labels.iterrows())
label2idx = {name: idx for idx, name in enumerate(classNames)}
train_label = [label2idx[name2label[os.path.basename(path)]] for path in train_path]

# # Run cross validation first
# print("\n" + "="*60)
# print("CROSS VALIDATION")
# print("="*60)
# acc, f1, cm = cross_val(CVFOLD, train_path, train_label)
# print(f"\nCross Validation Results:")
# print(f"Accuracy: {acc:.4f}")
# print(f"F1 Score: {f1:.4f}")
# print(f"Confusion Matrix:")
# print(cm)

# Train model and generate predictions
print("\n" + "="*60)
print("TRAINING AND PREDICTION")
print("="*60)

# Train and predict using fixed settings
output_file = train_and_predict_single_model(train_path, train_label, test_path)

print("\n" + "="*60)
print("EXECUTION COMPLETED")
print("="*60)
print(f"Generated output file: {output_file}")
print("The file is ready for submission!")
print("="*60)
