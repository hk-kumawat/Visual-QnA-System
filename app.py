import streamlit as st
from PIL import Image
from io import BytesIO
from transformers import BlipProcessor, BlipForConditionalGeneration, pipeline

# Set page layout to wide
st.set_page_config(page_title="Visual Question Answering", layout="wide")

# Load models with error handling
try:
    # Load BLIP pipeline for image captioning
    caption_pipeline = pipeline("image-to-text", model="Salesforce/blip-image-captioning-base")
    
    # Load BLIP processor and model for VQA
    vqa_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    vqa_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
except Exception as e:
    st.error(f"Error loading models: {e}")

# Function to generate image caption
def generate_caption(image):
    try:
        # Process image and generate caption
        img = Image.open(BytesIO(image)).convert("RGB")
        caption = caption_pipeline(img)[0]['generated_text']
        return caption
    except Exception as e:
        return f"Error generating caption: {e}"

# Function to answer a question about the image
def answer_question(image, question):
    try:
        # Process image and question for VQA
        img = Image.open(BytesIO(image)).convert("RGB")
        inputs = vqa_processor(img, question, return_tensors="pt")
        
        # Generate answer from model
        output = vqa_model.generate(**inputs)
        answer = vqa_processor.decode(output[0], skip_special_tokens=True)
        
        return answer
    except Exception as e:
        return f"Error generating answer: {e}"

# Set up the Streamlit app
st.title("🔍 Visual Question Answering 🖼️")
st.write("Upload an image and ask a question to get a detailed answer based on the content of the image.")

# Add custom CSS for styling
st.markdown(
    """
    <style>
    .title {
        text-align: center;
        font-size: 40px;
        color: #4CAF50;
        font-weight: bold;
        margin-bottom: 50px;
    }
    .centered {
        text-align: center;
        font-size: 20px;
        font-style: italic;
        color: #333;
        margin-top: 20px;
        line-height: 1.6;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Create columns for image upload and input fields
col1, col2 = st.columns(2)

# Image upload
with col1:
    uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Display the uploaded image
        st.image(uploaded_file, use_container_width=True)
        
        # Generate and display image caption centered below the image
        image_bytes = uploaded_file.getvalue()
        caption = generate_caption(image_bytes)
        st.markdown(f"<div class='centered'>{caption}</div>", unsafe_allow_html=True)
    else:
        caption = None  # Handle case where no image is uploaded

# Custom Question Input
with col2:
    # Allow user to type their question if an image is uploaded
    question = st.text_input("Ask a question about the image")
    
    # Button for prediction (only show if an image is uploaded and a question is asked)
    if uploaded_file and question:
        if st.button("Get Answer"):
            # Get the answer
            answer = answer_question(image_bytes, question)
            
            # Display the answer
            st.success(f"Answer: {answer}")

# Footer
st.markdown("---")
st.markdown(
    "<p style='text-align: center;'>Where <strong>Vision</strong> Meets <strong>Intelligence</strong> - A Creation by <strong>Harshal Kumawat</strong> 👁️🤖</p>",
    unsafe_allow_html=True
)
