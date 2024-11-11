import streamlit as st
from PIL import Image
from io import BytesIO
from transformers import BlipProcessor, BlipForConditionalGeneration

# Set page layout to wide
st.set_page_config(page_title="Visual Question Answering", layout="wide")

# Load BLIP model for both image captioning and question answering
blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

# Function to get a detailed answer to a question (using BLIP)
def get_answer(image, question):
    try:
        # Load and process the image
        img = Image.open(BytesIO(image)).convert("RGB")
        
        # Generate caption for the image first
        caption = generate_caption(image)
        if caption == "Unknown":
            return "Failed to generate caption for the image."

        # Prepare the input for the BLIP model with caption and question
        input_text = f"Caption: {caption} Question: {question}"

        # Generate the answer using BLIP model with max_new_tokens to control output length
        inputs = blip_processor(images=img, text=input_text, return_tensors="pt")

        # Generate answer with a specific number of tokens
        out = blip_model.generate(**inputs, max_new_tokens=50)  # Limit the generated answer length to 50 tokens
        answer = blip_processor.decode(out[0], skip_special_tokens=True)
        
        return answer
    except Exception as e:
        return f"Error: {str(e)}"

# Function to generate image caption using BLIP
def generate_caption(image):
    try:
        # Prepare image for captioning
        img = Image.open(BytesIO(image)).convert("RGB")
        # Generate caption using BLIP
        inputs = blip_processor(images=img, return_tensors="pt")
        out = blip_model.generate(**inputs, max_new_tokens=30)  # Limit caption length
        caption = blip_processor.decode(out[0], skip_special_tokens=True)
        
        return caption
    except Exception as e:
        return "Unknown"  # Return "Unknown" if caption generation fails

# Set up the Streamlit app
st.title("🔍 Visual Question Answering 🖼️")
st.write("Upload an image and ask a question to get an answer!")

# Add custom CSS for styling
st.markdown(
    """
    <style>
    .title { text-align: center; font-size: 40px; color: #4CAF50; font-weight: bold; margin-bottom: 50px; }
    .centered { text-align: center; font-size: 20px; font-style: italic; color: #333; margin-top: 20px; line-height: 1.6; }
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
        # Display the uploaded image with a fixed width (500px)
        st.image(uploaded_file, width=500)
        # Generate and display image caption centered below the image
        image_bytes = uploaded_file.getvalue()
        caption = generate_caption(image_bytes)
        st.markdown(f"<div class='centered'>{caption}</div>", unsafe_allow_html=True)

# Custom Question Input
with col2:
    # Store the selected question in session state to persist its value
    if 'question' not in st.session_state:
        st.session_state['question'] = ""

    st.markdown("---")
    # Allow user to type their own question
    question = st.text_input("Your question", value=st.session_state['question'])

    # Button for prediction (only show if a question is provided)
    if uploaded_file and question:
        if st.button("Predict Answer"):
            image_bytes = uploaded_file.getvalue()
            # Get the detailed answer
            answer = get_answer(image_bytes, question)
            # Display the answer using st.success()
            st.success(f"Answer: {answer}")
            # After showing the answer, reset question input field for new input
            st.session_state['question'] = ""  # Clear the 'Your question' box for next input

# Footer
st.markdown("---")
st.markdown(
    "<p style='text-align: center;'>Where <strong>Vision</strong> Meets <strong>Intelligence</strong> - A Creation by <strong>Harshal Kumawat</strong> 👁️🤖</p>",
    unsafe_allow_html=True
)
