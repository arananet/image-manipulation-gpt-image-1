import streamlit as st
import os
import io
import base64
from PIL import Image
import requests
import json
from dotenv import load_dotenv
import time
from openai import AzureOpenAI

# Load environment variables
load_dotenv()

# Azure OpenAI for Image Generation
IMAGEGEN_AOAI_RESOURCE = os.getenv("IMAGEGEN_AOAI_RESOURCE", "eduar-ma108754-westus3")
IMAGEGEN_DEPLOYMENT = os.getenv("IMAGEGEN_DEPLOYMENT", "gpt-image-1")
IMAGEGEN_AOAI_API_KEY = os.getenv("IMAGEGEN_AOAI_API_KEY", "")

# Azure OpenAI for LLM
LLM_AOAI_RESOURCE = os.getenv("LLM_AOAI_RESOURCE", "ai-trainpoc7039ai740971184368")
LLM_DEPLOYMENT = os.getenv("LLM_DEPLOYMENT", "gpt-4.1-mini")
LLM_AOAI_API_KEY = os.getenv("LLM_AOAI_API_KEY", "")

# Initialize Azure OpenAI client
aoai_client = AzureOpenAI(
    azure_endpoint=f"https://{IMAGEGEN_AOAI_RESOURCE}.openai.azure.com/",
    api_version="2025-04-01-preview",
    api_key=IMAGEGEN_AOAI_API_KEY,
)

# Set page configuration
st.set_page_config(
    page_title="Image Manipulation Example for gpt-image-1",
    page_icon="🎨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main {
        padding: 1rem;
    }
    .stApp {
        max-width: 100%;
    }
    .hero-container {
        background-color: #f1f3f9;
        padding: 2rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .stButton button {
        background-color: #0066CC;
        color: white;
        font-weight: bold;
        padding: 0.5rem 1rem;
        border-radius: 5px;
        border: none;
    }
    .stButton button:hover {
        background-color: #004C99;
    }
    .upload-section, .result-section, .control-section {
        padding: 1.5rem;
        background-color: white;
        border-radius: 8px;
        margin-bottom: 1rem;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
    }
    .footer {
        margin-top: 2rem;
        text-align: center;
        font-size: 0.8rem;
        color: #666;
    }
    .preset-title {
        font-weight: bold;
        margin-bottom: 5px;
    }
    .preset-button {
        margin: 5px;
    }
    .preset-section {
        margin-top: 1rem;
        padding: 1rem;
        background-color: #f8f9fa;
        border-radius: 5px;
    }
    @media (max-width: 768px) {
        .hero-container {
            padding: 1rem;
        }
    }
</style>
""", unsafe_allow_html=True)

def generate_image(prompt, image_path=None, size="1024x1024", n=1, quality="high"):
    """Generate or edit an image using Azure OpenAI's image generation service"""
    try:
        url = None
        headers = {"api-key": IMAGEGEN_AOAI_API_KEY}
        
        # Determine API endpoint and payload based on provided parameters
        if image_path:
            # Image editing
            url = f"https://{IMAGEGEN_AOAI_RESOURCE}.openai.azure.com/openai/deployments/{IMAGEGEN_DEPLOYMENT}/images/edits?api-version=2025-04-01-preview"
            files = {
                "image": open(image_path, "rb"),
            }
            data = {
                "prompt": prompt,
                "n": n,
                "size": size,
                "quality": quality
            }
        else:
            # Text-to-image generation using SDK
            result = aoai_client.images.generate(
                model=IMAGEGEN_DEPLOYMENT,
                prompt=prompt,
                n=n,
                quality=quality,
                size=size,
                output_format="png",
            )
            
            images_data = result.model_dump()["data"]
            image_list = []
            
            for img in images_data:
                if "b64_json" in img:
                    image_bytes = base64.b64decode(img["b64_json"])
                    image_list.append(Image.open(io.BytesIO(image_bytes)))
            
            return image_list
        
        # Send the request for edit
        if url:
            response = requests.post(url, headers=headers, files=files, data=data)
            response.raise_for_status()
            
            images_data = response.json()["data"]
            image_list = []
            
            for img in images_data:
                if "b64_json" in img:
                    image_bytes = base64.b64decode(img["b64_json"])
                    image_list.append(Image.open(io.BytesIO(image_bytes)))
            
            return image_list
            
    except Exception as e:
        st.error(f"Error generating image: {str(e)}")
        return None

def save_uploaded_file(uploaded_file):
    """Save an uploaded file to disk and return the file path"""
    try:
        file_path = f"temp_{uploaded_file.name}"
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        return file_path
    except Exception as e:
        st.error(f"Error saving file: {str(e)}")
        return None

def get_preset_prompts():
    """Return a list of preset prompts for common image manipulations"""
    return {
        "Color Changes": [
            {"name": "Make it blue", "prompt": "Change the color to a vibrant blue shade, maintaining all details and lighting"},
            {"name": "Make it red", "prompt": "Change the color to a rich red shade, maintaining all details and lighting"},
            {"name": "Vintage effect", "prompt": "Apply a warm, sepia-toned vintage effect to the image"},
            {"name": "Black & white", "prompt": "Convert to a high-contrast black and white image"},
            {"name": "Neon glow", "prompt": "Add a vibrant neon glow effect to the main subject"}
        ],
        "Background Changes": [
            {"name": "White background", "prompt": "Place the main subject on a clean white background"},
            {"name": "Gradient background", "prompt": "Replace the background with a subtle blue to purple gradient"},
            {"name": "Natural scene", "prompt": "Place the subject in a natural outdoor setting with soft lighting"},
            {"name": "Studio lighting", "prompt": "Apply professional studio lighting with a neutral gray backdrop"},
            {"name": "Remove background", "prompt": "Isolate the main subject on a transparent background"}
        ],
        "Style Transfers": [
            {"name": "Oil painting", "prompt": "Transform into an oil painting style while preserving the subject"},
            {"name": "Watercolor", "prompt": "Convert to a delicate watercolor painting style"},
            {"name": "Anime style", "prompt": "Transform into anime/manga illustration style"},
            {"name": "Pencil sketch", "prompt": "Convert to a detailed pencil sketch"},
            {"name": "Pop art", "prompt": "Transform into vibrant pop art style like Andy Warhol"}
        ],
        "Scene Changes": [
            {"name": "Sunset lighting", "prompt": "Add warm sunset lighting to the scene"},
            {"name": "Nighttime", "prompt": "Convert to a nighttime scene with appropriate lighting"},
            {"name": "Rainy effect", "prompt": "Add a light rain effect with water droplets and reflections"},
            {"name": "Snow scene", "prompt": "Add a light snowfall effect to the scene"},
            {"name": "Foggy atmosphere", "prompt": "Add a mysterious fog effect to the scene"}
        ]
    }

def main():
    """Main Streamlit app function"""
    # Header section
    st.title("🎨 Image Manipulation Example for gpt-image-1")
    st.markdown("Transform your images using gpt-image-1 model for editing and generation")
    
    # Sidebar controls
    with st.sidebar:
        st.header("Settings")
        
        image_size = st.selectbox(
            "Image Size",
            ["1024x1024", "1024x1536", "1536x1024"],
            index=0
        )
        
        quality = st.selectbox(
            "Quality",
            ["high", "medium", "low"],
            index=0
        )
        
        num_variations = st.slider("Number of Results", 1, 4, 1)
        
        st.markdown("---")
        st.subheader("Advanced Options")
        
        st.markdown("---")
        with st.expander("About This App"):
            st.markdown("""
            This app uses Azure OpenAI's gpt-image-1 image generation and editing capabilities to transform images.
            
            **Features:**
            - Image editing with text prompts
            - Text-to-image generation
            - Multiple output variations
            """)
    
    # Create two columns for the main content
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<div class="upload-section">', unsafe_allow_html=True)
        st.subheader("Upload Image")
        
        # Options for different modes
        mode = st.radio(
            "Select Mode",
            ["Image Editing", "Text to Image"],
            horizontal=True
        )
        
        # File uploader for image editing mode
        image_path = None
        
        if mode == "Image Editing":
            uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
            
            if uploaded_file is not None:
                image = Image.open(uploaded_file)
                st.image(image, caption="Uploaded Image", use_column_width=True)
                image_path = save_uploaded_file(uploaded_file)
        
        st.markdown('<div class="control-section">', unsafe_allow_html=True)
        st.subheader("Edit Instructions")
        
        # Text area for custom prompt
        custom_prompt = st.text_area(
            "Enter your editing instructions",
            help="Describe the changes you want to make to the image"
        )
        
        # Preset prompts section
        st.markdown('<div class="preset-section">', unsafe_allow_html=True)
        st.markdown("**Preset Prompts**")
        
        preset_prompts = get_preset_prompts()
        preset_category = st.selectbox("Category", list(preset_prompts.keys()))
        
        # Create buttons for each preset in the selected category
        cols = st.columns(2)
        for i, preset in enumerate(preset_prompts[preset_category]):
            col_idx = i % 2
            with cols[col_idx]:
                if st.button(preset["name"], key=f"preset_{preset_category}_{i}"):
                    # Set the text area value to the preset prompt
                    st.session_state.custom_prompt = preset["prompt"]
                    st.rerun()
        
        # Update text area if preset was selected
        if 'custom_prompt' in st.session_state:
            custom_prompt = st.session_state.custom_prompt
            
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Generate button
        generate_button = st.button(
            "Generate Image" if mode == "Text to Image" else "Transform Image", 
            type="primary",
            use_container_width=True
        )
        
        st.markdown('</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="result-section">', unsafe_allow_html=True)
        st.subheader("Results")
        
        if generate_button:
            # Validate inputs based on mode
            if mode == "Image Editing" and not image_path:
                st.error("Please upload an image first.")
            elif not custom_prompt and mode != "Text to Image":
                st.error("Please enter editing instructions.")
            else:
                # Generate image based on mode
                with st.spinner("Generating images..."):
                    if mode == "Text to Image":
                        prompt = custom_prompt if custom_prompt else "A beautiful landscape with mountains and a lake"
                        images = generate_image(
                            prompt=prompt,
                            size=image_size,
                            n=num_variations,
                            quality=quality
                        )
                    else:  # Image Editing
                        images = generate_image(
                            prompt=custom_prompt,
                            image_path=image_path,
                            size=image_size,
                            n=num_variations,
                            quality=quality
                        )
                    
                    if images:
                        # Store results in session state
                        st.session_state.result_images = images
                        st.session_state.processing_complete = True
        
        # Display results if available
        if 'processing_complete' in st.session_state and st.session_state.processing_complete:
            # Create tabs for multiple results if needed
            if len(st.session_state.result_images) > 1:
                tabs = st.tabs([f"Result {i+1}" for i in range(len(st.session_state.result_images))])
                
                for i, (tab, img) in enumerate(zip(tabs, st.session_state.result_images)):
                    with tab:
                        st.image(img, caption=f"Generated Result {i+1}", use_column_width=True)
                        
                        # Save image to buffer for download
                        buf = io.BytesIO()
                        img.save(buf, format="PNG")
                        byte_im = buf.getvalue()
                        
                        st.download_button(
                            label="Download Image",
                            data=byte_im,
                            file_name=f"generated_image_{i+1}.png",
                            mime="image/png"
                        )
            else:
                # Single image display
                st.image(st.session_state.result_images[0], caption="Generated Result", use_column_width=True)
                
                # Save image to buffer for download
                buf = io.BytesIO()
                st.session_state.result_images[0].save(buf, format="PNG")
                byte_im = buf.getvalue()
                
                st.download_button(
                    label="Download Image",
                    data=byte_im,
                    file_name="generated_image.png",
                    mime="image/png"
                )
        else:
            st.info("Your transformed images will appear here")
            
        st.markdown('</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()
