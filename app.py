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

def create_zip_file(images, original_filename):
    # Create a temporary directory
    with tempfile.TemporaryDirectory() as temp_dir:
        # Get timestamp for unique zip name
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        zip_filename = f"augmented_images_{timestamp}.zip"
        zip_path = os.path.join(temp_dir, zip_filename)
        
        # Create ZIP file
        with zipfile.ZipFile(zip_path, 'w') as zip_file:
            # Save original image
            original_name = f"original_{original_filename}"
            images['original'][0].save(os.path.join(temp_dir, original_name))
            zip_file.write(os.path.join(temp_dir, original_name), original_name)
            
            # Save augmented images
            for idx, img in enumerate(images['augmented']):
                filename = f"augmented_{idx+1}_{original_filename}"
                img.save(os.path.join(temp_dir, filename))
                zip_file.write(os.path.join(temp_dir, filename), filename)
        
        # Read the ZIP file
        with open(zip_path, 'rb') as f:
            return f.read(), zip_filename

def main():
    st.title("Image Data Augmentation App")
    st.write("Upload an image and generate augmented versions!")

    # File uploader
    uploaded_file = st.file_uploader("Choose an image file", type=['png', 'jpg', 'jpeg'])
    
    if uploaded_file is not None:
        # Read image
        image = Image.open(uploaded_file).convert('RGB')
        st.image(image, caption="Original Image", use_column_width=True)
        
        # Store original filename
        original_filename = uploaded_file.name
        
        # Augmentation parameters
        st.sidebar.header("Augmentation Parameters")
        num_augmentations = st.sidebar.slider("Number of augmentations to generate", 1, 10, 3)
        
        # Transform options
        rotate = st.sidebar.checkbox("Enable rotation", True)
        rotation_degrees = st.sidebar.slider("Max rotation degrees", 0, 180, 30)
        
        flip = st.sidebar.checkbox("Enable random flip", True)
        
        crop = st.sidebar.checkbox("Enable random crop", True)
        crop_height = st.sidebar.number_input("Crop height", min_value=10, value=image.size[1])
        crop_width = st.sidebar.number_input("Crop width", min_value=10, value=image.size[0])
        
        color_jitter = st.sidebar.checkbox("Enable color jitter", True)
        brightness = st.sidebar.slider("Brightness variation", 0.0, 1.0, 0.2)
        contrast = st.sidebar.slider("Contrast variation", 0.0, 1.0, 0.2)
        saturation = st.sidebar.slider("Saturation variation", 0.0, 1.0, 0.2)
        
        grayscale = st.sidebar.checkbox("Convert to grayscale")
        sepia = st.sidebar.checkbox("Apply sepia filter")
        
        if st.button("Generate Augmentations"):
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
                'saturation': saturation
            }
            
            transforms = get_transforms(params)
            
            # Store generated images
            generated_images = {
                'original': [image],
                'augmented': []
            }
            
            cols = st.columns(3)
            for idx in range(num_augmentations):
                augmented_img = transforms(image)
                
                if grayscale:
                    augmented_img = F.rgb_to_grayscale(F.to_tensor(augmented_img), 3)
                    augmented_img = F.to_pil_image(augmented_img)
                
                if sepia:
                    augmented_img = apply_sepia(augmented_img)
                
                # Store the augmented image
                generated_images['augmented'].append(augmented_img)
                
                col_idx = idx % 3
                cols[col_idx].image(augmented_img, caption=f"Augmentation {idx+1}", use_column_width=True)
            
            # Create download button for ZIP file
            zip_data, zip_filename = create_zip_file(generated_images, original_filename)
            st.download_button(
                label="Download all images as ZIP",
                data=zip_data,
                file_name=zip_filename,
                mime="application/zip"
            )

if __name__ == "__main__":
    main()