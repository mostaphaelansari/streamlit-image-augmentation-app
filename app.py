import streamlit as st
import torch
import torchvision.transforms as T
from torchvision.transforms import functional as F
from PIL import Image
import io
import numpy as np
import zipfile
import os
import tempfile
from datetime import datetime

def apply_sepia(img):
    img_array = np.array(img)
    sepia_matrix = np.array([
        [0.393, 0.769, 0.189],
        [0.349, 0.686, 0.168],
        [0.272, 0.534, 0.131]
    ])
    sepia_img = img_array.dot(sepia_matrix.T)
    sepia_img[np.where(sepia_img > 255)] = 255
    return Image.fromarray(sepia_img.astype(np.uint8))

def get_transforms(params):
    transforms = []
    
    if params['rotate']:
        transforms.append(T.RandomRotation(params['rotation_degrees']))
    
    if params['flip']:
        transforms.append(T.RandomHorizontalFlip(p=1.0))
    
    if params['crop']:
        transforms.append(T.RandomResizedCrop(
            size=(params['crop_height'], params['crop_width']),
            scale=(0.8, 1.0)
        ))
    
    if params['color_jitter']:
        transforms.append(T.ColorJitter(
            brightness=params['brightness'],
            contrast=params['contrast'],
            saturation=params['saturation']
        ))
        
    return T.Compose(transforms)

def create_zip_file(all_images):
    # Create a temporary directory
    with tempfile.TemporaryDirectory() as temp_dir:
        # Get timestamp for unique zip name
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        zip_filename = f"augmented_images_{timestamp}.zip"
        zip_path = os.path.join(temp_dir, zip_filename)
        
        # Create ZIP file
        with zipfile.ZipFile(zip_path, 'w') as zip_file:
            # Process each original image and its augmentations
            for img_idx, (original_filename, images) in enumerate(all_images.items()):
                # Create a directory for each original image
                base_dir = f"image_{img_idx + 1}_{os.path.splitext(original_filename)[0]}"
                full_base_dir = os.path.join(temp_dir, base_dir)
                os.makedirs(full_base_dir, exist_ok=True)
                
                # Save original image
                original_name = os.path.join(base_dir, f"original_{original_filename}")
                original_path = os.path.join(temp_dir, original_name)
                images['original'][0].save(original_path)
                zip_file.write(original_path, original_name)
                
                # Save augmented images
                for aug_idx, img in enumerate(images['augmented']):
                    aug_name = os.path.join(base_dir, f"augmented_{aug_idx+1}_{original_filename}")
                    aug_path = os.path.join(temp_dir, aug_name)
                    img.save(aug_path)
                    zip_file.write(aug_path, aug_name)
        
        # Read the ZIP file
        with open(zip_path, 'rb') as f:
            return f.read(), zip_filename

def process_image(image, params, num_augmentations):
    transforms = get_transforms(params)
    augmented_images = []
    
    for _ in range(num_augmentations):
        augmented_img = transforms(image)
        
        if params['grayscale']:
            augmented_img = F.rgb_to_grayscale(F.to_tensor(augmented_img), 3)
            augmented_img = F.to_pil_image(augmented_img)
        
        if params['sepia']:
            augmented_img = apply_sepia(augmented_img)
        
        augmented_images.append(augmented_img)
    
    return augmented_images

def main():
    st.title("Multi-Image Data Augmentation App")
    st.write("Upload multiple images and generate augmented versions for all of them!")

    # File uploader for multiple files
    uploaded_files = st.file_uploader("Choose image files", type=['png', 'jpg', 'jpeg'], accept_multiple_files=True)
    
    if uploaded_files:
        # Show number of uploaded images
        st.write(f"📁 {len(uploaded_files)} images uploaded")
        
        # Display original images in a grid
        cols_original = st.columns(min(3, len(uploaded_files)))
        for idx, uploaded_file in enumerate(uploaded_files):
            image = Image.open(uploaded_file).convert('RGB')
            cols_original[idx % 3].image(image, caption=f"Original: {uploaded_file.name}", use_column_width=True)
        
        # Augmentation parameters
        st.sidebar.header("Augmentation Parameters")
        num_augmentations = st.sidebar.slider("Number of augmentations per image", 1, 10, 3)
        
        # Transform options
        rotate = st.sidebar.checkbox("Enable rotation", True)
        rotation_degrees = st.sidebar.slider("Max rotation degrees", 0, 180, 30)
        
        flip = st.sidebar.checkbox("Enable random flip", True)
        
        crop = st.sidebar.checkbox("Enable random crop", True)
        
        color_jitter = st.sidebar.checkbox("Enable color jitter", True)
        brightness = st.sidebar.slider("Brightness variation", 0.0, 1.0, 0.2)
        contrast = st.sidebar.slider("Contrast variation", 0.0, 1.0, 0.2)
        saturation = st.sidebar.slider("Saturation variation", 0.0, 1.0, 0.2)
        
        grayscale = st.sidebar.checkbox("Convert to grayscale")
        sepia = st.sidebar.checkbox("Apply sepia filter")
        
        if st.button("Generate Augmentations for All Images"):
            # Dictionary to store all generated images
            all_generated_images = {}
            
            # Progress bar
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            try:
                # Process each image
                for idx, uploaded_file in enumerate(uploaded_files):
                    status_text.text(f"Processing image {idx + 1} of {len(uploaded_files)}: {uploaded_file.name}")
                    
                    image = Image.open(uploaded_file).convert('RGB')
                    crop_height = image.size[1]
                    crop_width = image.size[0]
                    
                    params = {
                        'rotate': rotate,
                        'rotation_degrees': rotation_degrees,
                        'flip': flip,
                        'crop': crop,
                        'crop_height': crop_height,
                        'crop_width': crop_width,
                        'color_jitter': color_jitter,
                        'brightness': brightness,
                        'contrast': contrast,
                        'saturation': saturation,
                        'grayscale': grayscale,
                        'sepia': sepia
                    }
                    
                    # Process the image
                    augmented_images = process_image(image, params, num_augmentations)
                    
                    # Store results
                    all_generated_images[uploaded_file.name] = {
                        'original': [image],
                        'augmented': augmented_images
                    }
                    
                    # Display augmentations for this image
                    st.subheader(f"Augmentations for {uploaded_file.name}")
                    cols = st.columns(3)
                    for aug_idx, aug_img in enumerate(augmented_images):
                        cols[aug_idx % 3].image(aug_img, caption=f"Augmentation {aug_idx+1}", use_column_width=True)
                    
                    # Update progress
                    progress_bar.progress((idx + 1) / len(uploaded_files))
                
                status_text.text("Processing complete! You can now download all images.")
                
                # Create download button for ZIP file
                zip_data, zip_filename = create_zip_file(all_generated_images)
                st.download_button(
                    label="Download all augmented images as ZIP",
                    data=zip_data,
                    file_name=zip_filename,
                    mime="application/zip"
                )
                
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
                status_text.text("Processing failed. Please try again.")

if __name__ == "__main__":
    main()