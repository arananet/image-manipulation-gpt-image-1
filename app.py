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
    page_title="Image Manipulation Studio",
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

def generate_image(prompt, image_path=None, mask_path=None, size="1024x1024", n=1, quality="high"):
    """Generate or edit an image using Azure OpenAI's image generation service"""
    try:
        if image_path:
            # Validate image format
            img = Image.open(image_path)
            if img.format != "PNG":
                img = img.convert("RGB")
                temp_path = f"temp_converted_{os.path.basename(image_path)}.png"
                img.save(temp_path, "PNG")
                image_path = temp_path
            
            # Prepare files and data for multipart/form-data
            files = {
                "image": (os.path.basename(image_path), open(image_path, "rb"), "image/png")
            }
            data = {
                "prompt": prompt,
                "model": "gpt-image-1",
                "size": size,
                "n": str(n),  # Convert to string for form-data
                "quality": quality
            }
            
            if mask_path:
                # Validate mask format
                mask_img = Image.open(mask_path)
                if mask_img.format != "PNG":
                    mask_img = mask_img.convert("RGB")
                    temp_mask_path = f"temp_converted_{os.path.basename(mask_path)}.png"
                    mask_img.save(temp_mask_path, "PNG")
                    mask_path = temp_mask_path
                
                files["mask"] = (os.path.basename(mask_path), open(mask_path, "rb"), "image/png")
            
            # Image editing or inpainting
            url = f"https://{IMAGEGEN_AOAI_RESOURCE}.openai.azure.com/openai/deployments/{IMAGEGEN_DEPLOYMENT}/images/edits?api-version=2025-04-01-preview"
            headers = {"api-key": IMAGEGEN_AOAI_API_KEY}
            response = requests.post(url, headers=headers, files=files, data=data)
            
            # Check for error and log details
            if response.status_code != 200:
                error_detail = response.json().get("error", {})
                st.error(f"API Error: {response.status_code} - {error_detail.get('message', 'No details provided')}")
                return None
            
            # Log the raw API response for debugging
            response_json = response.json()
            st.write("API Response:", response_json)
            
            # Check if 'data' exists and is not empty
            if "data" not in response_json or not response_json["data"]:
                st.error("No images returned by the API. 'data' field is empty or missing.")
                return None
            
            images_data = response_json["data"]
            image_list = []
            
            for idx, img in enumerate(images_data):
                if "b64_json" in img:
                    # Handle base64-encoded image data
                    try:
                        image_bytes = base64.b64decode(img["b64_json"])
                        image_list.append(Image.open(io.BytesIO(image_bytes)))
                    except Exception as e:
                        st.error(f"Failed to decode base64 image data for item {idx}: {str(e)}")
                        continue
                elif "url" in img:
                    # Handle URL-based image
                    try:
                        img_response = requests.get(img["url"], timeout=10)
                        img_response.raise_for_status()
                        image_list.append(Image.open(io.BytesIO(img_response.content)))
                    except requests.exceptions.RequestException as e:
                        st.error(f"Failed to download image from URL {img['url']}: {str(e)}")
                        continue
                else:
                    st.error(f"No 'b64_json' or 'url' found in API response data item {idx}.")
                    continue
            
            if not image_list:
                st.error("No images were successfully processed.")
                return None
            
            return image_list
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
            
            # Log the raw API response for debugging
            result_json = result.model_dump()
            st.write("API Response (Text-to-Image):", result_json)
            
            images_data = result_json["data"]
            if not images_data:
                st.error("No images returned by the API (Text-to-Image). 'data' field is empty.")
                return None
            
            image_list = []
            
            for idx, img in enumerate(images_data):
                if "b64_json" not in img:
                    st.error(f"No base64 data found in API response data item {idx} (Text-to-Image).")
                    continue
                image_bytes = base64.b64decode(img["b64_json"])
                image_list.append(Image.open(io.BytesIO(image_bytes)))
            
            if not image_list:
                st.error("No images were successfully processed (Text-to-Image).")
                return None
            
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

def validate_mask(mask_path, target_dimensions):
    """Validate, binarize, resize, and make white areas transparent in the mask"""
    try:
        mask = Image.open(mask_path).convert("L")  # Convert to grayscale
        # Binarize the mask: pixels < 128 become 0 (black), >= 128 become 255 (white)
        mask = mask.point(lambda p: 255 if p >= 128 else 0, mode="1")
        
        # Resize mask to match target dimensions
        if mask.size != target_dimensions:
            mask = mask.resize(target_dimensions, Image.Resampling.NEAREST)  # Use NEAREST to preserve binary values
        
        # Convert white areas to transparent (alpha=0)
        mask_rgba = Image.new("RGBA", mask.size, (0, 0, 0, 255))  # Start with fully opaque black
        mask_data = mask.getdata()
        new_data = [(0, 0, 0, 0) if p == 255 else (0, 0, 0, 255) for p in mask_data]  # White -> transparent, Black -> opaque
        mask_rgba.putdata(new_data)
        
        # Save the processed mask
        binarized_path = f"temp_binarized_{os.path.basename(mask_path)}"
        mask_rgba.save(binarized_path, "PNG")
        return binarized_path
    except Exception as e:
        st.error(f"Error processing mask: {str(e)}")
        return None

def get_preset_prompts():
    """Return a list of preset prompts for common image manipulations"""
    return {
        "Inpainting Changes": [
            {"name": "Replace with Tree", "prompt": "Replace the masked region with a detailed tree, blending naturally with the scene"},
            {"name": "Change to Blue Shirt", "prompt": "Change the masked region to a blue shirt, matching the lighting and style"},
            {"name": "Add Flower", "prompt": "Add a vibrant flower in the masked region, blending with the surroundings"},
            {"name": "Replace with Sky", "prompt": "Replace the masked region with a clear blue sky with fluffy clouds"},
            {"name": "Remove Object", "prompt": "Remove the masked region and fill it with a seamless background matching the surroundings"}
        ],
        "Style Transfers": [
            {"name": "Oil Painting", "prompt": "Transform the image into an oil painting style with rich textures and vibrant colors"},
            {"name": "Watercolor", "prompt": "Convert the image to a delicate watercolor painting with soft edges and translucent hues"},
            {"name": "Anime Style", "prompt": "Redraw the image in a Japanese anime style with bold outlines and expressive colors"},
            {"name": "Cyberpunk", "prompt": "Apply a cyberpunk aesthetic with neon colors and futuristic elements"},
            {"name": "Vintage Photo", "prompt": "Make the image look like a vintage photograph with sepia tones and grainy texture"}
        ],
        "Lighting Changes": [
            {"name": "Sunset Glow", "prompt": "Apply warm sunset lighting with golden hues and soft shadows"},
            {"name": "Moonlit Night", "prompt": "Change the lighting to a cool, moonlit night with blue tones and subtle glow"},
            {"name": "Studio Lighting", "prompt": "Use professional studio lighting with even illumination and minimal shadows"},
            {"name": "Dramatic Spotlight", "prompt": "Add a dramatic spotlight effect focusing on the main subject"},
            {"name": "Foggy Morning", "prompt": "Apply soft, diffused lighting like a foggy morning with muted colors"}
        ],
        "Composition Changes": [
            {"name": "Add Beach Ball", "prompt": "Add a colorful beach ball in the center of the image, blending naturally with the scene"},
            {"name": "Expand Background", "prompt": "Extend the background to include a wider landscape, matching the original style"},
            {"name": "Add People", "prompt": "Include a group of people in the background, interacting naturally with the environment"},
            {"name": "Remove Objects", "prompt": "Remove any distracting objects from the background, keeping the main subject intact"},
            {"name": "Change Setting", "prompt": "Place the main subject in a new setting, such as a bustling city street"}
        ],
        "Background Changes": [
            {"name": "White Background", "prompt": "Place the main subject on a clean white background"},
            {"name": "Nature Scene", "prompt": "Replace the background with a lush forest scene and soft natural light"},
            {"name": "Urban Skyline", "prompt": "Set the background to a modern city skyline at dusk"},
            {"name": "Abstract Gradient", "prompt": "Use a smooth blue-to-purple gradient background"},
            {"name": "Transparent", "prompt": "Isolate the main subject on a transparent background"}
        ]
    }

def main():
    """Main Streamlit app function"""
    # Header section
    st.title("🎨 Image Manipulation Example for gpt-image-1")
    st.markdown("Transform your images using gpt-image-1 model for editing, inpainting, and generation")
    
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
            - Image editing with text prompts (style, lighting, composition)
            - Inpainting with uploaded mask to modify specific regions
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
            ["Image Editing", "Inpainting (Mask)", "Text to Image"],
            horizontal=True
        )
        
        # File uploaders based on mode
        image_path = None
        mask_path = None
        
        if mode in ["Image Editing", "Inpainting (Mask)"]:
            uploaded_file = st.file_uploader(
                "Upload a base photo (e.g., product shot or scenery)",
                type=["jpg", "jpeg", "png"],
                help="Upload a JPG, JPEG, or PNG image to edit or inpaint. PNG recommended."
            )
            
            if uploaded_file is not None:
                image = Image.open(uploaded_file)
                st.image(image, caption=f"Uploaded Image ({image.size[0]}x{image.size[1]})", use_container_width=True)
                image_path = save_uploaded_file(uploaded_file)
            
            if mode == "Inpainting (Mask)":
                uploaded_mask = st.file_uploader(
                    "Upload a mask (transparent or white for areas to change, black for areas to preserve)",
                    type=["jpg", "jpeg", "png"],
                    help="Upload a PNG image where areas to modify should be transparent or white (converted to transparent automatically), and areas to preserve should be black. Must match the base image dimensions (e.g., 1024x683 for the current image). The mask will be resized if dimensions differ."
                )
                
                if uploaded_mask is not None and image_path:
                    mask_path = save_uploaded_file(uploaded_mask)
                    validated_mask_path = validate_mask(mask_path, target_dimensions=image.size)
                    if validated_mask_path:
                        mask_path = validated_mask_path
                        mask_img = Image.open(mask_path)
                        st.image(mask_img, caption=f"Processed Mask (Resized to {mask_img.size[0]}x{mask_img.size[1]})", use_container_width=True)
                        # Validate dimensions after resizing
                        if image.size != mask_img.size:
                            st.error(f"Dimension mismatch: Image is {image.size}, but mask is {mask_img.size}. Please ensure the mask matches the image dimensions.")
                            mask_path = None
                    else:
                        st.error("Invalid mask: Could not process the mask. Ensure it's a PNG with transparent or white areas for editing (pure white #FFFFFF will be converted to transparent) and pure black #000000 for preserved areas.")
                        mask_path = None
        
        st.markdown('<div class="control-section">', unsafe_allow_html=True)
        st.subheader("Edit Instructions")
        
        # Instructions for each mode
        if mode == "Image Editing":
            st.markdown(
                "Describe how to modify the entire image (e.g., 'Change to watercolor style,' 'Add sunset lighting,' 'Place a beach ball in the center'). "
                "Use preset prompts below or enter a custom prompt."
            )
        elif mode == "Inpainting (Mask)":
            st.markdown(
                "Describe what to replace the masked region with (e.g., 'Replace with a tree,' 'Change to a blue shirt'). "
                "Ensure the mask has transparent or white areas for regions to change and black for areas to preserve."
            )
        else:
            st.markdown(
                "Enter a description to generate a new image (e.g., 'A beautiful landscape with mountains and a lake')."
            )
        
        # Text area for custom prompt
        custom_prompt = st.text_area(
            "Enter your editing instructions",
            help="For Image Editing: Describe style, lighting, or composition changes. For Inpainting: Describe what to replace the masked region with. For Text to Image: Describe the new image."
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
            elif mode == "Inpainting (Mask)" and (not image_path or not mask_path):
                st.error("Please upload both an image and a valid mask.")
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
                    elif mode == "Inpainting (Mask)":
                        images = generate_image(
                            prompt=custom_prompt,
                            image_path=image_path,
                            mask_path=mask_path,
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
                        st.image(img, caption=f"Generated Result {i+1}", use_container_width=True)
                        
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
                st.image(st.session_state.result_images[0], caption="Generated Result", use_container_width=True)
                
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
    
    # Footer
    st.markdown('<div class="footer">', unsafe_allow_html=True)
    st.markdown("Developed by Eduardo Arana - info@arananet.net")
    st.markdown('</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()
