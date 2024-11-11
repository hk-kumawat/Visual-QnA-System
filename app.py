import streamlit as st
from PIL import Image
from transformers import pipeline, AutoProcessor, AutoModelForVisualQuestionAnswering
import torch
from io import BytesIO

# Set page layout to wide
st.set_page_config(page_title="Visual Question Answering", layout="wide")

# Load the image-to-text pipeline for captioning
caption_pipeline = pipeline("image-to-text", model="Salesforce/blip2-opt-6.7b")

# Load processor and model for Visual Question Answering (VQA)
vqa_processor = AutoProcessor.from_pretrained("Salesforce/blip2-opt-6.7b")
vqa_model = AutoModelForVisualQuestionAnswering.from_pretrained("Salesforce/blip2-opt-6.7b")

# Function to generate image caption
def generate_image_caption(image):
    # Generate caption for the image
    caption = caption_pipeline(image)[0]['generated_text']
    return caption

# Function to answer questions based on an image
def answer_question(image, question):
    # Process the image and question for VQA
    inputs = vqa_processor(images=image, text=question, return_tensors="pt").to(vqa_model.device)
    
    # Generate answer
    with torch.no_grad():
        outputs = vqa_model(**inputs)
    
    # Decode and return answer
    answer = vqa_processor.decode(outputs.logits.argmax(dim=-1)[0])
    return answer

# Set up the Streamlit app
st.title("🔍 Visual Question Answering 🖼️ ")
st.write("Upload an image and enter a question to get an answer!")

# Add custom CSS for styling
st.markdown(
    """
    <style>
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
        img = Image.open(BytesIO(image_bytes)).convert("RGB")
        caption = generate_image_caption(img)

        # Display the caption with previous formatting and centered
        st.markdown(f"<div class='centered'>{caption}</div>", unsafe_allow_html=True)

# Question Input
with col2:
    question = st.text_input("Ask a question about the image")

    # Button for prediction
    if uploaded_file and question:
        if st.button("Predict Answer"):
            # Get the answer
            answer = answer_question(img, question)

            # Display the answer using st.success()
            st.success(f"Answer: {answer}")

# Footer
st.markdown("---")
st.markdown(
    "<p style='text-align: center;'>Where <strong>Vision</strong> Meets <strong>Intelligence</strong> - A Creation by <strong>Harshal Kumawat</strong> 👁️🤖</p>",
    unsafe_allow_html=True
)
