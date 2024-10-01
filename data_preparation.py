import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as img
import cv2
import skimage
from sklearn.feature_extraction.image import extract_patches_2d as patch_ex

def prep_lr(HR_path: str, LR_path: str, HR_patch_path: str, LR_patch_path: str, SR_scale: int) -> None:
    """
    Prepares low-resolution (LR) images from high-resolution (HR) images, 
    applies Gaussian blur, downsampling, and noise addition. 
    It also extracts patches from HR and LR images.
    
    Args:
        HR_path (str): Path to the directory containing high-resolution images.
        LR_path (str): Path to save the generated low-resolution images.
        HR_patch_path (str): Path to save the HR image patches.
        LR_patch_path (str): Path to save the LR image patches.
        SR_scale (int): Scaling factor for downsampling.
    """
    HR_filenames = os.listdir(HR_path)

    for file in HR_filenames:
        HR = img.imread(os.path.join(HR_path, file))

        # Gaussian Blur
        gaussian_blurred = cv2.GaussianBlur(HR, (0, 0), 4.0)

        # Bicubic Downsampling
        dim = (HR.shape[1] // SR_scale, HR.shape[0] // SR_scale)
        bicubic_downsampled = cv2.resize(gaussian_blurred, dim, interpolation=cv2.INTER_CUBIC)

        # Adding Noise
        LR = skimage.util.random_noise(bicubic_downsampled)

        # Saving the LR Image
        plt.imsave(os.path.join(LR_path, file), LR, format='png')

        # Patch extraction
        LR_patches = patch_ex(LR, (64, 64), max_patches=2, random_state=23)
        plt.imsave(os.path.join(LR_patch_path, f'p1_{file}'), LR_patches[0], format='png')
        plt.imsave(os.path.join(LR_patch_path, f'p2_{file}'), LR_patches[1], format='png')

        HR_patches = patch_ex(HR, (64, 64), max_patches=2, random_state=23)
        plt.imsave(os.path.join(HR_patch_path, f'p1_{file}'), HR_patches[0], format='png')
        plt.imsave(os.path.join(HR_patch_path, f'p2_{file}'), HR_patches[1], format='png')

if __name__ == "__main__":
    # Example usage
    prep_lr(
        HR_path='/home/bastin/PROJECT-main/Data/Train/HR/',
        LR_path='/home/bastin/PROJECT-main/Data/Train/LR/',
        HR_patch_path='/home/bastin/PROJECT-main/Data/Train/Patch_HR/',
        LR_patch_path='/home/bastin/PROJECT-main/Data/Train/Patch_LR/',
        SR_scale=4
    )
