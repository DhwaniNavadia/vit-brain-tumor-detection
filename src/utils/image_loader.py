import numpy as np
import nibabel as nib
import torch
import os
from torchvision import transforms

def load_mri_slice(image_path, slice_index=None):
    """
    Loads a 2D slice from a 3D MRI image file (.nii.gz).
    Args:
        image_path (str): Path to the .nii.gz file.
        slice_index (int): Index of the slice to extract. If None, center slice is returned.
    Returns:
        np.ndarray: 2D array of the selected slice.
    """
    image_obj = nib.load(image_path)
    image_data = image_obj.get_fdata()
    
    if image_data.ndim != 3:
        raise ValueError(f"Expected 3D image, got shape: {image_data.shape}")
    
    # Use center slice if not specified
    if slice_index is None:
        slice_index = image_data.shape[2] // 2

    slice_2d = image_data[:, :, slice_index]
    
    # Normalize to 0-1
    slice_2d = (slice_2d - np.min(slice_2d)) / (np.max(slice_2d) - np.min(slice_2d) + 1e-5)
    
    return slice_2d.astype(np.float32)


def preprocess_slice(slice_2d, image_size=224):
    """
    Preprocesses the 2D MRI slice to make it ready for ViT input.
    Args:
        slice_2d (np.ndarray): Normalized 2D MRI slice.
        image_size (int): Target image size for the model.
    Returns:
        torch.Tensor: Preprocessed tensor [1, H, W]
    """
    to_tensor = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor()
    ])
    
    tensor = to_tensor(slice_2d)
    return tensor

'''''

✅ What this does:
Loads a .nii.gz file and extracts a single normalized 2D slice.

Preprocesses it to the shape and format suitable for ViT input.

'''''