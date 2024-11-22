from pathlib import Path
import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io
import zipfile
from io import BytesIO
import random
import torchvision.transforms as T
import torch

# Page setup
st.set_page_config(
    page_title='Data Augmentation Helper',
    layout="wide",
    initial_sidebar_state="expanded"
)

# Make it look nice
st.markdown("""
    <style>
    .main { padding: 2rem; }
    .stButton > button { width: 100%; }
    .big-font { font-size: 24px; }
    .stAlert { padding: 1rem; }
    </style>
""", unsafe_allow_html=True)

def apply_augmentations(image, params):
    """
    Apply all selected augmentations to the image
    """
    img_array = np.array(image)
    result = img_array.copy()
    
    # Basic transformations
    if params['rotate']:
        angle = random.uniform(-params['rotation_range'], params['rotation_range'])
        height, width = result.shape[:2]
        center = (width // 2, height // 2)
        matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        result = cv2.warpAffine(result, matrix, (width, height))
    
    if params['flip_h'] and random.random() > 0.5:
        result = cv2.flip(result, 1)
        
    if params['flip_v'] and random.random() > 0.5:
        result = cv2.flip(result, 0)
    
    # Color transformations
    if params['brightness']:
        factor = random.uniform(1-params['brightness_range'], 1+params['brightness_range'])
        result = cv2.convertScaleAbs(result, alpha=1, beta=factor*50)
    
    if params['contrast']:
        factor = random.uniform(1-params['contrast_range'], 1+params['contrast_range'])
        result = cv2.convertScaleAbs(result, alpha=factor)
    
    # Scale/Zoom
    if params['zoom']:
        scale = random.uniform(params['zoom_range'][0], params['zoom_range'][1])
        height, width = result.shape[:2]
        result = cv2.resize(result, (int(width * scale), int(height * scale)))
    
    # Crop to original size if image was scaled up
    if result.shape[0] > img_array.shape[0] or result.shape[1] > img_array.shape[1]:
        y = random.randint(0, max(0, result.shape[0] - img_array.shape[0]))
        x = random.randint(0, max(0, result.shape[1] - img_array.shape[1]))
        result = result[y:y+img_array.shape[0], x:x+img_array.shape[1]]
    
    # Add noise
    if params['noise']:
        noise = np.random.normal(0, params['noise_level'], result.shape)
        result = np.clip(result + noise, 0, 255).astype(np.uint8)
    
    # Blur
    if params['blur']:
        kernel_size = random.choice([3, 5, 7])
        result = cv2.GaussianBlur(result, (kernel_size, kernel_size), 0)
        
    return result

def create_augmented_images(image, params, num_images):
    """
    Create multiple augmented versions of the image
    """
    augmented_images = []
    for _ in range(num_images):
        aug_img = apply_augmentations(image, params)
        augmented_images.append(aug_img)
    return augmented_images

def save_to_zip(images):
    """
    Save images to zip file
    """
    zip_data = BytesIO()
    with zipfile.ZipFile(zip_data, 'w', zipfile.ZIP_DEFLATED) as zip_file:
        for i, img in enumerate(images):
            img_pil = Image.fromarray(img)
            img_byte = BytesIO()
            img_pil.save(img_byte, format='PNG')
            zip_file.writestr(f'augmented_{i+1}.png', img_byte.getvalue())
    return zip_data.getvalue()

def main():
    st.title('📸 Data Augmentation Helper')
    st.markdown("### Make many versions of your image for training!")
    
    # File upload
    st.sidebar.header("📤 Upload Your Image")
    uploaded_file = st.sidebar.file_uploader(
        "Choose an image",
        type=['png', 'jpg', 'jpeg']
    )
    
    if uploaded_file:
        try:
            # Load image
            image = Image.open(uploaded_file)
            img_array = np.array(image)
            
            # Display original
            col1, col2 = st.columns(2)
            with col1:
                st.header("Original Image")
                st.image(image, use_column_width=True)
            
            # Augmentation settings
            st.sidebar.header("🎨 Augmentation Settings")
            
            # Basic transformations
            with st.sidebar.expander("🔄 Rotation & Flips", expanded=True):
                params = {
                    'rotate': st.checkbox("Enable Rotation", True),
                    'rotation_range': st.slider("Max Rotation (degrees)", 0, 180, 45),
                    'flip_h': st.checkbox("Random Horizontal Flip", True),
                    'flip_v': st.checkbox("Random Vertical Flip", False)
                }
            
            # Color transformations
            with st.sidebar.expander("🎨 Color Changes", expanded=True):
                params.update({
                    'brightness': st.checkbox("Random Brightness", True),
                    'brightness_range': st.slider("Brightness Range", 0.0, 1.0, 0.2),
                    'contrast': st.checkbox("Random Contrast", True),
                    'contrast_range': st.slider("Contrast Range", 0.0, 1.0, 0.2)
                })
            
            # Advanced transformations
            with st.sidebar.expander("🔍 Advanced Options", expanded=True):
                params.update({
                    'zoom': st.checkbox("Random Zoom", True),
                    'zoom_range': (
                        st.slider("Zoom Range", 0.5, 2.0, (0.8, 1.2)),
                    ),
                    'noise': st.checkbox("Add Noise", False),
                    'noise_level': st.slider("Noise Level", 0, 50, 25),
                    'blur': st.checkbox("Random Blur", False)
                })
            
            # Number of images to generate
            num_images = st.sidebar.number_input(
                "Number of images to create",
                min_value=1,
                max_value=50,
                value=8
            )
            
            # Generate button
            if st.sidebar.button("🎲 Generate New Images"):
                with st.spinner("Creating new images..."):
                    augmented_images = create_augmented_images(image, params, num_images)
                    
                    # Show results
                    with col2:
                        st.header("New Images")
                        # Show images in a grid
                        for i in range(0, min(8, len(augmented_images)), 2):
                            cols = st.columns(2)
                            for j in range(2):
                                if i + j < len(augmented_images):
                                    cols[j].image(augmented_images[i + j], use_column_width=True)
                    
                    # Download button
                    zip_file = save_to_zip(augmented_images)
                    st.sidebar.download_button(
                        "📥 Download All Images",
                        zip_file,
                        "augmented_images.zip",
                        "application/zip"
                    )
                    
                    st.success(f"Created {len(augmented_images)} new images!")
            
        except Exception as e:
            st.error(f"Oops! Something went wrong: {str(e)}")
            
    else:
        st.info("👆 Start by uploading an image!")

if __name__ == "__main__":
    main()