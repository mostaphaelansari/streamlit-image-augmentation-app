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

# Set page configuration
st.set_page_config(
    page_title='Multi-Image Data Augmentation App',
    page_icon='🖼️',
    layout='wide',
    initial_sidebar_state='expanded'
)

# Apply custom CSS
st.markdown("""
    <style>
        .stButton>button {
            width: 100%;
        }
        .stProgress .st-bo {
            background-color: #00ff00;
        }
        .success-message {
            padding: 1rem;
            border-radius: 0.5rem;
            background-color: #d4edda;
            color: #155724;
            margin: 1rem 0;
        }
        .error-message {
            padding: 1rem;
            border-radius: 0.5rem;
            background-color: #f8d7da;
            color: #721c24;
            margin: 1rem 0;
        }
    </style>
""", unsafe_allow_html=True)

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

@st.cache_data
def get_transforms(params):
    transforms = []
    
    if params['rotate']:
        transforms.append(T.RandomRotation(params['rotation_degrees']))
    
    if params['flip']:
        transforms.append(T.RandomHorizontalFlip(p=1.0))
    
    if params['vertical_flip'] and params['flip']:
        transforms.append(T.RandomVerticalFlip(p=1.0))
    
    if params['crop']:
        transforms.append(T.RandomResizedCrop(
            size=(params['crop_height'], params['crop_width']),
            scale=(params['crop_scale_min'], params['crop_scale_max'])
        ))
    
    if params['color_jitter']:
        transforms.append(T.ColorJitter(
            brightness=params['brightness'],
            contrast=params['contrast'],
            saturation=params['saturation'],
            hue=params['hue']
        ))
        
    if params['gaussian_blur']:
        transforms.append(T.GaussianBlur(
            kernel_size=(5, 5),
            sigma=params['blur_sigma']
        ))
        
    return T.Compose(transforms)

def create_zip_file(all_images, include_originals=True):
    with tempfile.TemporaryDirectory() as temp_dir:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        zip_filename = f"augmented_images_{timestamp}.zip"
        zip_path = os.path.join(temp_dir, zip_filename)
        
        with zipfile.ZipFile(zip_path, 'w', compression=zipfile.ZIP_DEFLATED) as zip_file:
            for img_idx, (original_filename, images) in enumerate(all_images.items()):
                base_dir = f"image_{img_idx + 1}_{os.path.splitext(original_filename)[0]}"
                full_base_dir = os.path.join(temp_dir, base_dir)
                os.makedirs(full_base_dir, exist_ok=True)
                
                if include_originals:
                    original_name = os.path.join(base_dir, f"original_{original_filename}")
                    original_path = os.path.join(temp_dir, original_name)
                    images['original'][0].save(original_path, quality=95)
                    zip_file.write(original_path, original_name)
                
                for aug_idx, img in enumerate(images['augmented']):
                    aug_name = os.path.join(base_dir, f"augmented_{aug_idx+1}_{original_filename}")
                    aug_path = os.path.join(temp_dir, aug_name)
                    img.save(aug_path, quality=95)
                    zip_file.write(aug_path, aug_name)
        
        with open(zip_path, 'rb') as f:
            return f.read(), zip_filename

@st.cache_data
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
    st.title("🖼️ Multi-Image Data Augmentation App")
    st.markdown("""
    Upload multiple images and generate augmented versions with various transformations.
    Perfect for creating diverse training datasets!
    """)

    # Sidebar configuration
    with st.sidebar:
        st.header("⚙️ Augmentation Parameters")
        
        # Basic parameters
        num_augmentations = st.slider("Number of augmentations per image", 1, 20, 3)
        include_originals = st.checkbox("Include original images in ZIP", value=True)
        
        st.subheader("Transform Options")
        
        # Geometric transforms
        with st.expander("Geometric Transforms", expanded=True):
            rotate = st.checkbox("Enable rotation", True)
            rotation_degrees = st.slider("Max rotation degrees", 0, 180, 30)
            
            flip = st.checkbox("Enable horizontal flip", True)
            vertical_flip = st.checkbox("Enable vertical flip", False)
            
            crop = st.checkbox("Enable random crop", True)
            crop_scale_min = st.slider("Minimum crop scale", 0.5, 1.0, 0.8)
            crop_scale_max = st.slider("Maximum crop scale", 0.5, 1.0, 1.0)
        
        # Color transforms
        with st.expander("Color Transforms", expanded=True):
            color_jitter = st.checkbox("Enable color jitter", True)
            brightness = st.slider("Brightness variation", 0.0, 1.0, 0.2)
            contrast = st.slider("Contrast variation", 0.0, 1.0, 0.2)
            saturation = st.slider("Saturation variation", 0.0, 1.0, 0.2)
            hue = st.slider("Hue variation", 0.0, 0.5, 0.1)
            
            grayscale = st.checkbox("Convert to grayscale")
            sepia = st.checkbox("Apply sepia filter")
            
            gaussian_blur = st.checkbox("Apply Gaussian blur")
            blur_sigma = st.slider("Blur intensity", 0.1, 5.0, 1.0)

    # File uploader
    uploaded_files = st.file_uploader(
        "Choose image files",
        type=['png', 'jpg', 'jpeg'],
        accept_multiple_files=True,
        help="You can select multiple files at once"
    )
    
    if uploaded_files:
        st.write(f"📁 {len(uploaded_files)} images uploaded")
        
        # Display original images in a grid
        cols_original = st.columns(min(3, len(uploaded_files)))
        for idx, uploaded_file in enumerate(uploaded_files):
            image = Image.open(uploaded_file).convert('RGB')
            cols_original[idx % 3].image(image, caption=f"Original: {uploaded_file.name}", use_column_width=True)
        
        if st.button("🚀 Generate Augmentations for All Images"):
            all_generated_images = {}
            
            # Create a container for the progress bar and status
            with st.container():
                progress_bar = st.progress(0)
                status_text = st.empty()
            
            try:
                for idx, uploaded_file in enumerate(uploaded_files):
                    status_text.text(f"🔄 Processing image {idx + 1} of {len(uploaded_files)}: {uploaded_file.name}")
                    
                    image = Image.open(uploaded_file).convert('RGB')
                    crop_height = image.size[1]
                    crop_width = image.size[0]
                    
                    params = {
                        'rotate': rotate,
                        'rotation_degrees': rotation_degrees,
                        'flip': flip,
                        'vertical_flip': vertical_flip,
                        'crop': crop,
                        'crop_height': crop_height,
                        'crop_width': crop_width,
                        'crop_scale_min': crop_scale_min,
                        'crop_scale_max': crop_scale_max,
                        'color_jitter': color_jitter,
                        'brightness': brightness,
                        'contrast': contrast,
                        'saturation': saturation,
                        'hue': hue,
                        'grayscale': grayscale,
                        'sepia': sepia,
                        'gaussian_blur': gaussian_blur,
                        'blur_sigma': blur_sigma
                    }
                    
                    augmented_images = process_image(image, params, num_augmentations)
                    all_generated_images[uploaded_file.name] = {
                        'original': [image],
                        'augmented': augmented_images
                    }
                    
                    st.subheader(f"Augmentations for {uploaded_file.name}")
                    cols = st.columns(3)
                    for aug_idx, aug_img in enumerate(augmented_images):
                        cols[aug_idx % 3].image(aug_img, caption=f"Augmentation {aug_idx+1}", use_column_width=True)
                    
                    progress_bar.progress((idx + 1) / len(uploaded_files))
                
                st.markdown(
                    '<div class="success-message">✅ Processing complete! You can now download all images.</div>',
                    unsafe_allow_html=True
                )
                
                zip_data, zip_filename = create_zip_file(all_generated_images, include_originals)
                st.download_button(
                    label="📦 Download all augmented images as ZIP",
                    data=zip_data,
                    file_name=zip_filename,
                    mime="application/zip"
                )
                
            except Exception as e:
                st.markdown(
                    f'<div class="error-message">❌ An error occurred: {str(e)}</div>',
                    unsafe_allow_html=True
                )
                status_text.text("Processing failed. Please try again.")

if __name__ == "__main__":
    main()