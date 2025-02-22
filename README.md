# Image Data Augmentation App 🖼️

A powerful and user-friendly Streamlit application for performing various image augmentation techniques on multiple images simultaneously. This tool is perfect for machine learning practitioners, data scientists, and anyone working with image datasets who needs to expand their training data through augmentation.

![Python](https://img.shields.io/badge/Python-3.7+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.0+-red.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-1.8+-green.svg)
![License](https://img.shields.io/badge/license-MIT-blue.svg)

- **Hosting Link**:
  - Verify that the hosting link ([demo_link] ([https://data-image-augmentation.streamlit.app/](https://data-augmentation-images.streamlit.app/))) is correct and active.

## ✨ Features

- **Multiple Image Processing**: Upload and process multiple images simultaneously
- **Various Augmentation Techniques**:
  - Rotation (adjustable degrees)
  - Random horizontal flip
  - Random crop with resizing
  - Color jitter (brightness, contrast, saturation)
  - Grayscale conversion
  - Sepia filter
- **Batch Processing**: Process all uploaded images with the same augmentation parameters
- **Real-time Preview**: See augmented versions immediately in the web interface
- **Progress Tracking**: Visual progress bar during processing
- **Bulk Download**: Download all original and augmented images in an organized ZIP file
- **User-Friendly Interface**: Easy-to-use sidebar controls for augmentation parameters

## 🚀 Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/image-augmentation-app.git
cd image-augmentation-app
```

2. Create and activate a virtual environment (optional but recommended):
```bash
# Windows
python -m venv env
env\Scripts\activate

# Linux/Mac
python3 -m venv env
source env/bin/activate
```

3. Install required packages:
```bash
pip install -r requirements.txt
```

## 📦 Requirements

Create a `requirements.txt` file with the following dependencies:
```
streamlit>=1.0.0
torch>=1.8.0
torchvision>=0.9.0
Pillow>=8.0.0
numpy>=1.19.0
```

## 💻 Usage

1. Start the Streamlit app:
```bash
streamlit run app.py
```

2. Open your web browser and navigate to the URL shown in your terminal (typically `http://localhost:8501`)

3. Use the app:
   - Upload one or multiple images using the file uploader
   - Adjust augmentation parameters in the sidebar
   - Click "Generate Augmentations for All Images"
   - Preview the results
   - Download all images as a ZIP file

## 📁 Output Structure

The downloaded ZIP file will be organized as follows:
```
augmented_images_TIMESTAMP.zip
├── image_1_filename1/
│   ├── original_filename1.jpg
│   ├── augmented_1_filename1.jpg
│   ├── augmented_2_filename1.jpg
│   └── ...
├── image_2_filename2/
│   ├── original_filename2.jpg
│   ├── augmented_1_filename2.jpg
│   └── ...
└── ...
```

## 🎛️ Available Parameters

| Parameter | Description | Range |
|-----------|-------------|--------|
| Number of augmentations | Number of augmented versions per image | 1-10 |
| Rotation degrees | Maximum rotation angle | 0-180° |
| Brightness variation | Brightness adjustment range | 0.0-1.0 |
| Contrast variation | Contrast adjustment range | 0.0-1.0 |
| Saturation variation | Saturation adjustment range | 0.0-1.0 |

## 🤝 Contributing

Contributions are welcome! Here's how you can help:

1. Fork the repository
2. Create a new branch (`git checkout -b feature/improvement`)
3. Make your changes
4. Commit your changes (`git commit -am 'Add new feature'`)
5. Push to the branch (`git push origin feature/improvement`)
6. Create a Pull Request

## 🐛 Bug Reports

If you find a bug, please create an issue with:
- A clear description of the bug
- Steps to reproduce
- Expected behavior
- Screenshots (if applicable)

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Built with [Streamlit](https://streamlit.io/)
- Image processing powered by [PyTorch](https://pytorch.org/) and [torchvision](https://pytorch.org/vision/stable/index.html)
- Image manipulation using [Pillow](https://python-pillow.org/)

## 📞 Contact

- EL ANSARI Mostapha - [elansarimostapha011@gmail.com]

---

Made with ❤️ by [Mostapha EL ANSARI]
