import streamlit as st
from PIL import Image
from io import BytesIO
from transformers import ViltProcessor, ViltForQuestionAnswering, BlipProcessor, BlipForConditionalGeneration

# Set page layout to wide
st.set_page_config(page_title="Visual Q&A", layout="wide")

# Load VQA model
processor = ViltProcessor.from_pretrained("dandelin/vilt-b32-finetuned-vqa")
model = ViltForQuestionAnswering.from_pretrained("dandelin/vilt-b32-finetuned-vqa")

# Load image captioning model (BLIP)
blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

# Function to get the answer to a question
def get_answer(image, text):
    try:
        # Load and process the image
        img = Image.open(BytesIO(image)).convert("RGB")

        # Prepare inputs for VQA
        encoding = processor(img, text, return_tensors="pt")

        # Forward pass
        outputs = model(**encoding)
        logits = outputs.logits
        idx = logits.argmax(-1).item()
        answer = model.config.id2label[idx]

        return answer

    except Exception as e:
        return str(e)

# Function to generate image caption
def generate_caption(image):
    try:
        # Prepare image for captioning
        img = Image.open(BytesIO(image)).convert("RGB")
        
        # Generate caption
        inputs = blip_processor(images=img, return_tensors="pt")
        out = blip_model.generate(**inputs)
        caption = blip_processor.decode(out[0], skip_special_tokens=True)

        return caption

    except Exception as e:
        return str(e)

# Set up the Streamlit app
st.title("🌟 Visual Question Answering 🌟")
st.write("Upload an image and choose a suggested question to get an answer!")

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
        margin-top: 20px;
        font-size: 20px;
        color: #333;
        font-style: italic;
    }
    .card {
        background-color: #f0f8ff;
        border-radius: 10px;
        box-shadow: 0px 4px 12px rgba(0, 0, 0, 0.1);
        padding: 20px;
    }
    .question-input {
        margin-bottom: 20px;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border-radius: 10px;
        padding: 12px;
        font-size: 16px;
        box-shadow: 0px 4px 12px rgba(0, 0, 0, 0.1);
    }
    .stButton>button:hover {
        background-color: #45a049;
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

# Suggested Question Input
with col2:
    # List of suggested questions (you can dynamically generate these based on the image if desired)
    suggested_questions = [
        "What is in the image?",
        "What is the object in the image?",
        "Describe the image.",
        "What is the person doing in the image?"
    ]
    
    question = st.selectbox("Choose a suggested question", suggested_questions)

    # Button for prediction
    if uploaded_file and question:
        if st.button("Predict Answer"):
            image_bytes = uploaded_file.getvalue()

            # Get the answer
            answer = get_answer(image_bytes, question)

            # Display the answer in a stylish box
            st.markdown(
                f"""
                <div class="card">
                    <h4 style="text-align: center;">Answer:</h4>
                    <p style="text-align: center; font-size: 18px; color: #333;">{answer}</p>
                </div>
                """,
                unsafe_allow_html=True
            )
