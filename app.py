import streamlit as st
from PIL import Image
from io import BytesIO
from transformers import ViltProcessor, ViltForQuestionAnswering, BlipProcessor, BlipForConditionalGeneration

# Set page layout to wide
st.set_page_config(layout="wide")

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
st.title("Visual Question Answering with Follow-up Questions")
st.write("Upload an image and enter a question to get an answer.")

# Create columns for image upload and input fields
col1, col2 = st.columns(2)

# Image upload
with col1:
    uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Display the uploaded image
        st.image(uploaded_file, use_container_width=True)

        # Generate and display image caption
        image_bytes = uploaded_file.getvalue()
        caption = generate_caption(image_bytes)
        st.write("Image Caption: " + caption)

# Question input
with col2:
    question = st.text_input("Question")

    # Store previous answer to allow follow-up questions
    if 'previous_answer' not in st.session_state:
        st.session_state.previous_answer = ""

    # Process the image and question when both are provided
    if uploaded_file and question:
        if st.button("Ask Question"):
            image_bytes = uploaded_file.getvalue()

            # Get the answer
            answer = get_answer(image_bytes, question)

            # Display the answer
            st.success("Answer: " + answer)

            # Store the current answer for follow-up questions
            st.session_state.previous_answer = answer

    # Allow follow-up question based on previous answer
    if st.session_state.previous_answer:
        follow_up_question = st.text_input("Follow-up Question (optional)", key="follow_up")

        if follow_up_question:
            if st.button("Ask Follow-up Question"):
                # Get the follow-up answer
                follow_up_answer = get_answer(image_bytes, follow_up_question)
                st.success("Follow-up Answer: " + follow_up_answer)
                st.session_state.previous_answer = follow_up_answer
