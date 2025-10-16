import cv2
import numpy as np
import streamlit as st
from PIL import Image

# Helper Functions for Image Loading and Noise Generation

def load_image(image_file):
    """
    Opens an image file, converts it to RGB format, and returns it as a NumPy array.
    """
    img = Image.open(image_file)
    return np.array(img.convert('RGB'))

def add_gaussian_noise(image, mean=0, std=25):
    """
    Adds Gaussian noise to an image.
    """
    row, col, ch = image.shape
    gauss = np.random.normal(mean, std, (row, col, ch))
    gauss = gauss.reshape(row, col, ch)
    
    noisy_image = image + gauss
    # Clip values to be in the valid range [0, 255]
    noisy_image = np.clip(noisy_image, 0, 255).astype(np.uint8)
    
    return noisy_image

# Core Image Processing Functions

def get_sharpened_images(image):
    """
    Applies various sharpening filters to an image and returns a dictionary of results.
    """
    # 1. Kernel-based sharpening
    kernel = np.array([[-1, -1, -1], 
                       [-1,  9, -1],
                       [-1, -1, -1]])
    sharpened_kernel = cv2.filter2D(image, -1, kernel)
    
    # 2. Unsharp masking technique using addWeighted
    gaussian_blur_5x5 = cv2.GaussianBlur(image, (5, 5), 1)
    unsharp_mask_medium = cv2.addWeighted(image, 2.5, gaussian_blur_5x5, -1.5, 0)
    unsharp_mask_heavy = cv2.addWeighted(image, 3.5, gaussian_blur_5x5, -2.5, 0)
    
    gaussian_blur_7x7 = cv2.GaussianBlur(image, (7, 7), 3)
    unsharp_mask_high_radius = cv2.addWeighted(image, 6.5, gaussian_blur_7x7, -5.5, 0)

    return {
        "Kernel": sharpened_kernel,
        "Unsharp Mask (Medium)": unsharp_mask_medium,
        "Unsharp Mask (Heavy)": unsharp_mask_heavy,
        "Unsharp Mask (High Radius)": unsharp_mask_high_radius
    }

def get_edge_detected_images(image):
    """
    Applies various edge detection algorithms and returns a dictionary of results.
    """
    gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    
    # 1. Sobel Edge Detection
    sobel_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)  
    sobel_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)  
    sobel_combined = cv2.convertScaleAbs(np.sqrt(sobel_x**2 + sobel_y**2))

    # 2. Prewitt Edge Detection 
    kernel_prewitt_x = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
    kernel_prewitt_y = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]])
    prewitt_x = cv2.filter2D(image, -1, kernel_prewitt_x)
    prewitt_y = cv2.filter2D(image, -1, kernel_prewitt_y)
    prewitt_combined = cv2.add(prewitt_x, prewitt_y)

    # 3. Canny Edge Detection (works best on grayscale)
    canny_edges = cv2.Canny(gray_image, 100, 200)

    return {
        "Sobel": sobel_combined,
        "Prewitt": prewitt_combined,
        "Canny": canny_edges
    }
    
# Streamlit UI Display Functions

def create_denoising_showcase(original_image):
    """
    Creates the denoising section in the Streamlit app.
    """
    st.header("1. Image Denoising")
    
    noisy_img = add_gaussian_noise(original_image)
    
    # Apply different denoising filters
    mean_denoised = cv2.blur(noisy_img, ksize=(5, 5))
    median_denoised = cv2.medianBlur(noisy_img, ksize=5)
    bilateral_denoised = cv2.bilateralFilter(noisy_img, d=9, sigmaColor=75, sigmaSpace=75) 
    
    # Display images in columns
    cols = st.columns(5)
    cols[0].image(original_image, caption="Original")
    cols[1].image(noisy_img, caption="Noisy (Gaussian)")
    cols[2].image(mean_denoised, caption="Denoised (Mean)")
    cols[3].image(median_denoised, caption="Denoised (Median)")
    cols[4].image(bilateral_denoised, caption='Denoised (Bilateral)')

def create_sharpening_showcase(original_image):
    """
    Creates the sharpening section in the Streamlit app.
    """
    st.header("2. Image Sharpening")
    
    sharpened_images = get_sharpened_images(original_image)
    
    # Display images in columns
    cols = st.columns(5)
    cols[0].image(original_image, caption="Original")
    
    # Display each sharpened image from the dictionary
    for i, (method, img) in enumerate(sharpened_images.items()):
        cols[i+1].image(img, caption=method)

def create_edge_detection_showcase(original_image):
    """
    Creates the edge detection section in the Streamlit app.
    """
    st.header("3. Edge Detection")
    
    edge_images = get_edge_detected_images(original_image)
    
    # Display images in columns
    cols = st.columns(4)
    cols[0].image(original_image, caption="Original")
    cols[1].image(edge_images["Sobel"], caption="Sobel")
    cols[2].image(edge_images["Prewitt"], caption="Prewitt")
    cols[3].image(edge_images["Canny"], caption="Canny")
    
# Main Application Logic

def main():
    """
    The main function to run the Streamlit application.
    """
    st.set_page_config(layout="wide")
    st.title("Image Enhancement Techniques")

    uploaded_file = st.file_uploader(
        "Choose an image file", 
        type=["jpg", "png", "jpeg"]
    )

    if uploaded_file is not None:
        original_img = load_image(uploaded_file)
        
        # Create each showcase section
        st.divider()
        create_denoising_showcase(original_img)
        st.divider()
        create_sharpening_showcase(original_img)
        st.divider()
        create_edge_detection_showcase(original_img)

if __name__ == '__main__':
    main()