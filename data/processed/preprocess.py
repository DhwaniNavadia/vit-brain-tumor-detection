# src/preprocessing/preprocess.py

import os
import nibabel as nib
import numpy as np
import torch
from torchvision import transforms
from tqdm import tqdm

# Constants
MODALITIES = ['flair', 't1', 't1ce', 't2']
TARGET_SHAPE = (128, 128, 128)

def load_nifti_file(filepath):
    return nib.load(filepath).get_fdata()

def normalize(volume):
    volume = (volume - np.mean(volume)) / np.std(volume)
    return np.clip(volume, -5, 5)

def resize_volume(volume, shape=TARGET_SHAPE):
    from scipy.ndimage import zoom
    factors = [s/o for s, o in zip(shape, volume.shape)]
    return zoom(volume, factors, order=1)

def preprocess_patient(patient_path):
    modalities = []
    for mod in MODALITIES:
        file = [f for f in os.listdir(patient_path) if mod in f.lower() and f.endswith('.nii.gz')]
        if not file:
            raise FileNotFoundError(f"{mod} modality missing in {patient_path}")
        filepath = os.path.join(patient_path, file[0])
        volume = load_nifti_file(filepath)
        volume = normalize(volume)
        volume = resize_volume(volume)
        modalities.append(volume)

    # Shape: (C, D, H, W)
    tensor = torch.tensor(np.stack(modalities)).float()
    return tensor

def preprocess_all(raw_dir, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    patient_dirs = [os.path.join(raw_dir, d) for d in os.listdir(raw_dir) if os.path.isdir(os.path.join(raw_dir, d))]

    for path in tqdm(patient_dirs, desc="Preprocessing"):
        try:
            tensor = preprocess_patient(path)
            patient_id = os.path.basename(path)
            torch.save(tensor, os.path.join(save_dir, f"{patient_id}.pt"))
        except Exception as e:
            print(f"Skipping {path} due to error: {e}")

if __name__ == "__main__":
    raw_data_path = "data/raw/BraTS2021_Training_Data"
    save_path = "data/processed"
    preprocess_all(raw_data_path, save_path)
