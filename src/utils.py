import os
import numpy as np
import pandas as pd
import pickle
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
import ast
from tqdm import tqdm
import wfdb
from scipy.signal import butter, filtfilt

DATASET_PATH = '/content/drive/MyDrive/datathon/ptb-xl/'
CSV_PATH = os.path.join(DATASET_PATH, 'ptbxl_database.csv')
RECORDS_PATH = DATASET_PATH 
NOPY_PATH = './ecg_npy_100'

os.makedirs(NOPY_PATH, exist_ok=True)

df = pd.read_csv(CSV_PATH)

# Convert signal 100Hz to .npy
for idx, row in tqdm(df.iterrows(), total=len(df), desc="Converting ECGs"):
    try:
        filename = row['filename_lr']
        ecg_id = str(row['ecg_id']).zfill(5)
        record_path = os.path.join(RECORDS_PATH, filename)

        npy_file = os.path.join(NOPY_PATH, f"{ecg_id}_lr.npy")
        if os.path.exists(npy_file):
            continue 

        record = wfdb.rdrecord(record_path)
        ecg_data = record.p_signal.T 

        np.save(npy_file, ecg_data)

    except Exception as e:
        print(f"Error on {row['ecg_id']}: {e}")

def parse_scp_codes(code_str):
    try:
        return ast.literal_eval(code_str)
    except:
        return {}

df['scp_codes_dict'] = df['scp_codes'].apply(parse_scp_codes)

with open(os.path.join(DATASET_PATH, 'all_labels.pkl'), 'rb') as f:
    all_labels = pickle.load(f)

assert 'strat_fold' in df.columns, "strat_fold not found."
train_df = df[df['strat_fold'].isin(range(1, 9))]
val_df   = df[df['strat_fold'] == 9]
test_df  = df[df['strat_fold'] == 10]

def bandpass_filter(signal, lowcut=0.5, highcut=40, fs=100, order=4):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, signal, axis=1)


class PTBXLDataset(Dataset):
    def __init__(self, dataframe, all_labels, npy_dir, duration=10, sampling_rate=100):
        self.df = dataframe.copy()
        self.all_labels = all_labels
        self.npy_dir = npy_dir
        self.n_samples = duration * sampling_rate
        self.mlb = MultiLabelBinarizer(classes=all_labels)
        self.mlb.fit([all_labels])
        self.sampling_rate = sampling_rate

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        ecg_id = row.name
        npy_path = os.path.join(self.npy_dir, f"{str(ecg_id).zfill(5)}_lr.npy")

        try:
            signal = np.load(npy_path)  # shape: (12, 1000)
            signal = bandpass_filter(signal, lowcut=0.5, highcut=40, fs=self.sampling_rate)
            signal = self.normalize_signal(signal)

            labels = list(row['scp_codes_dict'].keys()) if row['scp_codes_dict'] else []
            label_vector = self.mlb.transform([labels])[0].astype(np.float32)

            return {
                'signal': torch.tensor(signal, dtype=torch.float32),
                'labels': torch.tensor(label_vector, dtype=torch.float32),
                'record_id': ecg_id
            }

        except Exception as e:
            print(f" Error loading {npy_path}: {e}")
            return {
                'signal': torch.zeros(12, self.n_samples),
                'labels': torch.zeros(len(self.all_labels)),
                'record_id': ecg_id
            }

    def normalize_signal(self, signal):
        norm = np.zeros_like(signal)
        for i in range(signal.shape[0]):
            lead = signal[i]
            mean, std = np.mean(lead), np.std(lead)
            norm[i] = (lead - mean) / std if std > 0 else lead
        return norm


total = len(df)
print(f"Train: {len(train_df)} ({len(train_df)/total*100:.1f}%)")
print(f"Val:   {len(val_df)} ({len(val_df)/total*100:.1f}%)")
print(f"Test:  {len(test_df)} ({len(test_df)/total*100:.1f}%)")
print(f"Total labels: {len(all_labels)}")

batch_size = 32
train_dataset = PTBXLDataset(train_df, all_labels, npy_dir=NOPY_PATH)
val_dataset   = PTBXLDataset(val_df, all_labels, npy_dir=NOPY_PATH)
test_dataset  = PTBXLDataset(test_df, all_labels, npy_dir=NOPY_PATH)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,  num_workers=2)
val_loader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False, num_workers=2)
test_loader  = DataLoader(test_dataset,  batch_size=batch_size, shuffle=False, num_workers=2)