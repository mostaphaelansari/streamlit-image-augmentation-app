import streamlit as st
import numpy as np
from PIL import Image
import io
import zipfile
import albumentations as A
import cv2
import base64
from io import BytesIO
import os
import tempfile

def get_augmentation_pipeline():
    """Define the augmentation pipeline"""
    return A.Compose([
        A.RandomRotate90(p=0.5),
        A.Flip(p=0.5),
        A.Transpose(p=0.5),
        A.OneOf([
            A.IAAAdditiveGaussianNoise(),
            A.GaussNoise(),
        ], p=0.2),
        A.OneOf([
            A.MotionBlur(p=0.2),
            A.MedianBlur(blur_limit=3, p=0.1),
            A.Blur(blur_limit=3, p=0.1),
        ], p=0.2),
        A.OneOf([
            A.OpticalDistortion(p=0.3),
            A.GridDistortion(p=0.1),
            A.IAAPiecewiseAffine(p=0.3),
        ], p=0.2),
        A.OneOf([
            A.CLAHE(clip_limit=2),
            A.IAASharpen(),
            A.IAAEmboss(),
            A.RandomBrightnessContrast(),
        ], p=0.3),
        A.HueSaturationValue(p=0.3),
    ])

def augment_image(image, pipeline):
    """Apply augmentation to a single image"""
    augmented = pipeline(image=np.array(image))
    return Image.fromarray(augmented['image'])

def create_download_link(img, filename):
    """Create a download link for a single image"""
    buffered = BytesIO()
    img.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    href = f'<a href="data:file/png;base64,{img_str}" download="{filename}">Download {filename}</a>'
    return href

def main():
    st.title("Image Augmentation App")
    st.write("Upload images or a ZIP file to apply augmentations")

    # Sidebar controls
    st.sidebar.header("Augmentation Settings")
    num_augmentations = st.sidebar.slider("Number of augmentations per image", 1, 10, 3)

    # File uploader
    uploaded_file = st.file_uploader("Choose files", type=['png', 'jpg', 'jpeg', 'zip'], accept_multiple_files=False)

    if uploaded_file is not None:
        # Create augmentation pipeline
        pipeline = get_augmentation_pipeline()

        if uploaded_file.type == "application/zip":
            # Handle ZIP file
            with zipfile.ZipFile(uploaded_file) as zip_file:
                # Create a temporary directory to store augmented images
                with tempfile.TemporaryDirectory() as temp_dir:
                    augmented_images = []
                    
                    # Process each image in the ZIP file
                    for filename in zip_file.namelist():
                        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                            with zip_file.open(filename) as file:
                                img = Image.open(io.BytesIO(file.read())).convert('RGB')
                                
                                # Create augmentations
                                for i in range(num_augmentations):
                                    aug_img = augment_image(img, pipeline)
                                    aug_filename = f"aug_{i}_{filename}"
                                    aug_path = os.path.join(temp_dir, aug_filename)
                                    aug_img.save(aug_path)
                                    augmented_images.append(aug_path)
                    
                    # Create a ZIP file with augmented images
                    memory_file = BytesIO()
                    with zipfile.ZipFile(memory_file, 'w') as zf:
                        for aug_path in augmented_images:
                            zf.write(aug_path, os.path.basename(aug_path))
                    
                    # Create download button for ZIP
                    st.download_button(
                        label="Download Augmented Images (ZIP)",
                        data=memory_file.getvalue(),
                        file_name="augmented_images.zip",
                        mime="application/zip"
                    )
        else:
            # Handle single image
            image = Image.open(uploaded_file).convert('RGB')
            st.image(image, caption="Original Image", use_column_width=True)
            
            # Create and display augmentations
            st.write("Augmented Images:")
            cols = st.columns(3)
            for i in range(num_augmentations):
                aug_img = augment_image(image, pipeline)
                cols[i % 3].image(aug_img, caption=f"Augmentation {i+1}", use_column_width=True)
                
                # Create download link for each augmented image
                filename = f"augmented_{i+1}.png"
                cols[i % 3].markdown(create_download_link(aug_img, filename), unsafe_allow_html=True)

if __name__ == "__main__":
    main()