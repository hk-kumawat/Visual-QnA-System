import streamlit as st
from PIL import Image
from io import BytesIO
from transformers import BlipProcessor, BlipForConditionalGeneration
import gc
import torch

# Set up the page layout
st.set_page_config(page_title="Advanced Visual Question Answering", layout="wide")

# Load the VQA model and processor with error handling
try:
    # Use a smaller VQA model for reduced memory consumption
    vqa_processor = BlipProcessor.from_pretrained("Salesforce/blip-vqa-small")
    vqa_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-vqa-small")
    vqa_model = vqa_model.half()  # Use half precision to reduce memory usage
except Exception as e:
    st.error(f"Error loading models: {e}")

# Function to answer a question based on an image
def answer_question(image, question):
    try:
        # Load and prepare the image
        img = Image.open(BytesIO(image)).convert("RGB")
        
        # Process image and question as inputs
        inputs = vqa_processor(images=img, text=question, return_tensors="pt")
        
        # Generate a response with an increased maximum token length
        output = vqa_model.generate(**inputs, max_length=20, num_beams=3, early_stopping=True)
        
        # Decode and return the answer
        answer = vqa_processor.decode(output[0], skip_special_tokens=True)
        
        # Clear memory after processing
        gc.collect()
        torch.cuda.empty_cache()

        return answer
    except Exception as e:
        return f"Error generating answer: {e}"

# Streamlit interface setup
st.title("🔍 Advanced Visual Question Answering 🖼️")
st.write("Upload an image and ask a question. The app will provide an answer based on the image content.")

# Image upload and question input
col1, col2 = st.columns(2)

# Image upload section
with col1:
    uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        # Display the uploaded image
        st.image(uploaded_file, use_container_width=True)
        image_bytes = uploaded_file.getvalue()
    else:
        image_bytes = None  # Ensure image_bytes is defined even if no image is uploaded

# Question input section
with col2:
    if image_bytes:
        question = st.text_input("Ask a question about the image")
        
        # Trigger the model to answer upon button click
        if question and st.button("Get Answer"):
            # Generate and display answer
            answer = answer_question(image_bytes, question)
            st.success(f"Answer: {answer}")

# Footer
st.markdown("---")
st.markdown(
    "<p style='text-align: center;'>Where <strong>Vision</strong> Meets <strong>Intelligence</strong> - A Creation by <strong>Harshal Kumawat</strong> 👁️🤖</p>",
    unsafe_allow_html=True
)
