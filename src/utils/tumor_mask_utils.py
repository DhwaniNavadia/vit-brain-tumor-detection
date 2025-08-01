import numpy as np
import nibabel as nib
from skimage.measure import label, regionprops

def load_segmentation_mask(mask_path, slice_index=None):
    """
    Load a 2D slice from a 3D segmentation mask (.nii.gz).
    Args:
        mask_path (str): Path to the segmentation .nii.gz file.
        slice_index (int): Index of the slice to extract. If None, uses the center slice.
    Returns:
        np.ndarray: 2D binary mask of tumor.
    """
    mask_obj = nib.load(mask_path)
    mask_data = mask_obj.get_fdata()
    
    if mask_data.ndim != 3:
        raise ValueError(f"Expected 3D mask, got shape: {mask_data.shape}")
    
    if slice_index is None:
        slice_index = mask_data.shape[2] // 2

    mask_slice = mask_data[:, :, slice_index]

    # Convert to binary: consider non-zero values as tumor
    binary_mask = (mask_slice > 0).astype(np.uint8)
    
    return binary_mask


def get_bounding_box(mask_slice):
    """
    Returns bounding box coordinates for the largest tumor region in the mask.
    Args:
        mask_slice (np.ndarray): 2D binary mask.
    Returns:
        tuple: (min_row, min_col, max_row, max_col) or None if no tumor found.
    """
    labeled = label(mask_slice)
    regions = regionprops(labeled)

    if not regions:
        return None

    largest_region = max(regions, key=lambda r: r.area)
    minr, minc, maxr, maxc = largest_region.bbox
    return (minr, minc, maxr, maxc)


def calculate_tumor_area(mask_slice):
    """
    Calculate total tumor area (number of pixels).
    """
    return np.sum(mask_slice)




"""

✅ This gives you:
load_segmentation_mask: Gets 2D tumor mask slice.

get_bounding_box: Extracts coordinates of tumor region.

calculate_tumor_area: Measures tumor size in pixels.

"""