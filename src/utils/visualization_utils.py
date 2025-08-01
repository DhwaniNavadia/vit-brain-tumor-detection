import matplotlib.pyplot as plt
import numpy as np

def show_mri_and_mask(mri_slice, mask_slice, title="MRI with Tumor Mask", figsize=(6, 6)):
    """
    Display the MRI slice and overlay the tumor mask.
    """
    plt.figure(figsize=figsize)
    plt.imshow(mri_slice, cmap='gray')
    plt.imshow(mask_slice, cmap='Reds', alpha=0.4)
    plt.title(title)
    plt.axis('off')
    plt.show()


def show_attention_map(image, attention_map, title="Attention Map Overlay", alpha=0.4):
    """
    Overlay attention map on image.
    """
    attention_resized = np.array(attention_map)
    if attention_resized.shape != image.shape:
        raise ValueError("Attention map and image must be the same shape")

    plt.figure(figsize=(6, 6))
    plt.imshow(image, cmap='gray')
    plt.imshow(attention_resized, cmap='jet', alpha=alpha)
    plt.title(title)
    plt.axis('off')
    plt.show()


def draw_bounding_box(image, bbox, title="Tumor Bounding Box", figsize=(6, 6)):
    """
    Draw bounding box on the image.
    """
    from matplotlib.patches import Rectangle

    fig, ax = plt.subplots(1, figsize=figsize)
    ax.imshow(image, cmap='gray')
    
    if bbox:
        minr, minc, maxr, maxc = bbox
        rect = Rectangle((minc, minr), maxc - minc, maxr - minr,
                         linewidth=2, edgecolor='lime', facecolor='none')
        ax.add_patch(rect)

    ax.set_title(title)
    ax.axis('off')
    plt.show()



"""

This gives you:
show_mri_and_mask – Tumor highlighted on MRI

show_attention_map – Attention overlay on MRI

draw_bounding_box – Bounding box around tumor (from mask)

"""