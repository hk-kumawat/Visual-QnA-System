import streamlit as st
from PIL import Image
import requests
from io import BytesIO
from transformers import ViltProcessor, ViltForQuestionAnswering
from transformers import pipeline

# Set page layout to wide
st.set_page_config(layout="wide")

# Load the Vilt model and processor
processor = ViltProcessor.from_pretrained("dandelin/vilt-b32-finetuned-vqa")
model = ViltForQuestionAnswering.from_pretrained("dandelin/vilt-b32-finetuned-vqa")

# Load an object detection model
object_detection = pipeline("object-detection")

# Function to generate potential questions based on detected objects
def generate_question_suggestions(image):
    detected_objects = object_detection(image)
    objects = list(set(obj['label'] for obj in detected_objects))
    
    question_templates = [
        "What is the {} in the image?",
        "How many {} are there?",
        "What color is the {}?",
        "Is there a {} in the picture?",
        "Where is the {} located?"
    ]
    
    suggestions = []
    for obj in objects:
        for template in question_templates:
            suggestions.append(template.format(obj))
    
    return suggestions

# Function to get the answer for a given question
def get_answer(image, text):
    try:
        # Load and process the image
        img = Image.open(BytesIO(image)).convert("RGB")

        # Prepare inputs
        encoding = processor(img, text, return_tensors="pt")

        # Forward pass
        outputs = model(**encoding)
        logits = outputs.logits
        idx = logits.argmax(-1).item()
        answer = model.config.id2label[idx]

        return answer

    except Exception as e:
        return str(e)

# Streamlit app layout
st.title("Visual Question Answering")
st.write("Upload an image, select or enter a question to get an answer.")

# Image upload
uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

# Process the uploaded image
if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Generate question suggestions
    with st.spinner("Analyzing image content for question suggestions..."):
        suggestions = generate_question_suggestions(image)
    
    # Display a dropdown with question suggestions
    selected_question = st.selectbox("Suggested Questions", options=[""] + suggestions)
    
    # User question input
    user_question = st.text_input("Or enter your own question")
    
    # Determine the final question to use
    question = user_question if user_question else selected_question

    # Process the image and question when both are provided
    if question and st.button("Ask Question"):
        # Convert image to byte array for model processing
        image_byte_array = BytesIO()
        image.save(image_byte_array, format='JPEG')
        image_bytes = image_byte_array.getvalue()
        
        # Get the answer
        answer = get_answer(image_bytes, question)
        
        # Display the answer
        st.success("Answer: " + answer)
