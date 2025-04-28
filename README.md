### Image Manipulation Studio

A Streamlit-based web application that leverages Azure OpenAI's gpt-image-1 model for image generation and editing. Users can generate images from text prompts or edit existing images using custom or preset instructions.

## Features

- Text-to-Image Generation: Create images from text prompts.
- Image Editing: Upload an image and apply transformations based on text instructions.
- Preset Prompts: Choose from categorized preset prompts for common edits (e.g., color changes, background changes, style transfers, scene changes).
- Customizable Settings: Adjust image size, quality, and number of output variations.
- Downloadable Results: Save generated or edited images as PNG files.

## Prerequisites

- Python 3.8 or higher
- Azure OpenAI subscription with access to the gpt-image-1 model
- Environment variables configured in a .env file

## Installation

1. Clone the Repository:
   ```
   git clone https://github.com/arananet/image-manipulation-gpt-image-1
   cd image-manipulation-gpt-image-1
   ```

3. Install Dependencies:
   ```
   Create a virtual environment and install the required packages:
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt

   streamlit
   python-dotenv
   pillow
   requests
   openai

   ```

5. Configure Environment Variables:
   ```
   Create a .env file in the project root with the following variables:
   IMAGEGEN_AOAI_RESOURCE=your-resource-name
   IMAGEGEN_DEPLOYMENT=your-deployment-name
   IMAGEGEN_AOAI_API_KEY=your-api-key
   LLM_AOAI_RESOURCE=your-llm-resource-name
   LLM_DEPLOYMENT=your-llm-deployment-name
   LLM_AOAI_API_KEY=your-llm-api-key
   ```

7. Run the Application:
   ```
   streamlit run app.py

   The app will be available at http://localhost:8501.
   ```

## Usage

1. Select Mode:
   - Choose Text to Image to generate an image from a text prompt.
   - Choose Image Editing to upload and edit an existing image.

2. Upload Image (Image Editing Mode):
   - Upload a JPG, JPEG, or PNG image.

3. Enter Instructions:
   - Provide a custom prompt in the text area, or select a preset prompt from categories like Color Changes, Background Changes, Style Transfers, or Scene Changes.

4. Configure Settings:
   - In the sidebar, adjust the image size, quality, and number of results (1-4).

5. Generate/Transform:
   - Click the "Generate Image" or "Transform Image" button to process the request.
   - Results will appear in the right column, with options to download each image.

## Project Structure
```
image-manipulation-studio/
├── app.py              # Main Streamlit application
├── .env                # Environment variables (not tracked)
├── requirements.txt    # Python dependencies
└── README.txt          # Project documentation
```

## Notes

- The application uses Azure OpenAI's image generation API, which requires a valid subscription and API key.
- Temporary files are created for uploaded images and deleted after processing.
- The app supports images in PNG format for output.
- The LLM-related environment variables are included but not used in the current implementation.

## Author

Developed by Eduardo Arana

## License

This project is licensed under the MIT License.
