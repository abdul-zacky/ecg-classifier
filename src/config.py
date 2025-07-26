import os
import torch

# Path dataset
DATASET_PATH = '/content/drive/MyDrive/datathon/ptb-xl/'
NPY_PATH = os.path.join(DATASET_PATH, 'ecg_npy_100')

# Hyperparameter
BATCH_SIZE = 32
NUM_EPOCHS = 25
LEARNING_RATE = 1e-3

# Direktori output
os.makedirs('models', exist_ok=True)
os.makedirs('logs', exist_ok=True)