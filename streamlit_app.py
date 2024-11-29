import cv2
import numpy as np
import streamlit as st
from PIL import Image

def apply_smooth_pen_style_effect(image, scale_factor=4):
    """Applies a smoother pen-style effect with refined line connections."""
    image = np.array(image)
    height, width = image.shape[:2]
    upscaled_image = cv2.resize(image, (width * scale_factor, height * scale_factor), interpolation=cv2.INTER_CUBIC)
    
    # Convert to grayscale
    gray = cv2.cvtColor(upscaled_image, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur for smoother gradients
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)
    
    # Edge detection using Canny with optimized thresholds
    edges = cv2.Canny(blurred, 40, 120)
    
    # Morphological closing to connect edge segments
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    refined_edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)
    
    # Anti-aliasing: Apply Gaussian blur to soften edges
    smoothed_edges = cv2.GaussianBlur(refined_edges, (5, 5), 0)
    
    # Invert to get black lines on a white background
    inverted_edges = cv2.bitwise_not(smoothed_edges)
    
    # Convert to binary for clean black-and-white output
    _, final_output = cv2.threshold(inverted_edges, 240, 255, cv2.THRESH_BINARY)
    
    return final_output

# Streamlit app
st.title("Smooth Pen-Style Effect on Images")

# Upload image
uploaded_file = st.file_uploader("Upload an image file", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display uploaded image
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
    
    # Convert uploaded file to PIL Image
    image = Image.open(uploaded_file)
    
    # Apply the effect
    st.write("Processing the image...")
    output_image = apply_smooth_pen_style_effect(image)
    
    # Display output image
    st.image(output_image, caption="Processed Image", use_column_width=True, channels="GRAY")
    
    # Allow download of processed image
    output_pil = Image.fromarray(output_image)
    output_pil.save("processed_image.jpg")
    with open("processed_image.jpg", "rb") as file:
        btn = st.download_button(
            label="Download Processed Image",
            data=file,
            file_name="processed_image.jpg",
            mime="image/jpeg"
        )
