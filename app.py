import streamlit as st
from PIL import Image
from io import BytesIO
from transformers import BlipProcessor, BlipForConditionalGeneration

# Set page layout to wide
st.set_page_config(page_title="Visual Question Answering", layout="wide")

# Function to load the model with error handling
def load_model(model_name):
    try:
        processor = BlipProcessor.from_pretrained(model_name)
        model = BlipForConditionalGeneration.from_pretrained(model_name)
        return processor, model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None

# Load BLIP-2 model (BLIP's enhanced version for question answering)
blip_processor, blip_model = load_model("Salesforce/blip2-opt-6.7b")

# Ensure the model is loaded successfully
if not blip_processor or not blip_model:
    st.stop()

# Function to get the answer to a question using BLIP-2
def get_answer(image, text):
    try:
        # Check if image or question is None or empty
        if image is None:
            return "No image provided."
        if not text or text.strip() == "":
            return "No question provided."

        # Load and process the image
        img = Image.open(BytesIO(image)).convert("RGB")

        # Prepare the image and question for BLIP-2
        inputs = blip_processor(images=img, text=text, return_tensors="pt")

        if inputs is None:
            return "Error: Failed to create inputs for the BLIP model."

        # Generate an answer based on the image and question
        out = blip_model.generate(**inputs)
        answer = blip_processor.decode(out[0], skip_special_tokens=True)

        return answer

    except Exception as e:
        return f"Error occurred: {str(e)}"

# Set up the Streamlit app
st.title("🔍 Visual Question Answering 🖼️")
st.write("Upload an image and ask a question to get a detailed answer!")

# Create columns for image upload and input fields
col1, col2 = st.columns(2)

# Image upload
with col1:
    uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Display the uploaded image
        st.image(uploaded_file, use_container_width=True)

        # Store image bytes for processing
        image_bytes = uploaded_file.getvalue()

        # Suggested Questions - you can expand or customize the question set
        suggested_questions = [
            "What is in the image?",
            "What is happening in the image?",
            "Who or what is in the image?",
            "What is the main subject of the image?"
        ]
    else:
        suggested_questions = []  # Handle case where no image is uploaded

# Question Input
with col2:
    # Store the selected question in session state to persist its value
    if 'question' not in st.session_state:
        st.session_state['question'] = ""

    # Allow user to type their own question
    question = st.text_input("Ask a question about the image", value=st.session_state['question'])

    # Button for prediction (only show if question is input)
    if uploaded_file and question:
        if st.button("Get Answer"):
            # Get the detailed answer from BLIP-2
            answer = get_answer(image_bytes, question)

            # Display the answer
            st.success(f"Answer: {answer}")

            # After showing the answer, reset the question input field for new input
            st.session_state['question'] = ""  # Clear the 'Your question' box for next input

# Footer
st.markdown("---")
st.markdown(
    "<p style='text-align: center;'>Where <strong>Vision</strong> Meets <strong>Intelligence</strong> - A Creation by <strong>Harshal Kumawat</strong> 👁️🤖</p>",
    unsafe_allow_html=True
)

