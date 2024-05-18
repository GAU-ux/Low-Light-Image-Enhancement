import os
import cv2
from skimage.metrics import structural_similarity as ssim

# Function to calculate PSNR
def calculate_psnr(original_image, distorted_image):
    original_gray = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
    distorted_gray = cv2.cvtColor(distorted_image, cv2.COLOR_BGR2GRAY)
    psnr_value = cv2.PSNR(original_gray, distorted_gray)
    return psnr_value

# Function to calculate SSIM
def calculate_ssim(original_image, distorted_image):
    original_gray = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
    distorted_gray = cv2.cvtColor(distorted_image, cv2.COLOR_BGR2GRAY)
    ssim_value = ssim(original_gray, distorted_gray)
    return ssim_value

# Path to the folders containing the original and distorted images
original_folder_path = './testing_MEF/MEF/'
distorted_folder_path = './testing_MEF/MEF/'
# testing_MEF\MEF_Result\MEF\I

# Get list of image filenames in the original folder
original_image_files = os.listdir(original_folder_path)

# Iterate through original images and find corresponding distorted images
for original_image_file in original_image_files:
    original_image_path = os.path.join(original_folder_path, original_image_file)
    
    # Extract distorted image filename
    distorted_image_file = original_image_file.replace('original_', 'distorted_')
    distorted_image_path = os.path.join(distorted_folder_path, distorted_image_file)
    
    # Check if distorted image exists
    if os.path.isfile(distorted_image_path):
        # Load original and distorted images
        original_image = cv2.imread(original_image_path)
        distorted_image = cv2.imread(distorted_image_path)
        
        # Calculate PSNR and SSIM
        psnr_value = calculate_psnr(original_image, distorted_image)
        ssim_value = calculate_ssim(original_image, distorted_image)
        
        print(f"Original image: {original_image_file}, Distorted image: {distorted_image_file}")
        print(f"PSNR value: {psnr_value}")
        print(f"SSIM value: {ssim_value}")
        print()
    else:
        print(f"Distorted image for {original_image_file} not found.")
