import streamlit as st
from PIL import Image
from io import BytesIO
from transformers import ViltProcessor, ViltForQuestionAnswering, BlipProcessor, BlipForConditionalGeneration

# Set page layout to wide
st.set_page_config(page_title="Visual Question Answering", layout="wide")

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
st.title("🔍 Visual Question Answering 🖼️")
st.write("Upload an image and ask a question to get an answer!")

# Help Section
st.markdown(
    """
    <div style="background-color: #f1f1f1; padding: 10px; border-radius: 10px;">
    <h3 style="color: #333;">📘 Help: How to use this app</h3>
    <p style="color: #333;">Welcome to the Visual Question Answering app! Here's how you can make the most of it:</p>
    <ul style="color: #333;">
        <li><strong>Step 1:</strong> Upload an image by clicking on the "Upload Image" button.</li>
        <li><strong>Step 2:</strong> After the image is uploaded, type in your question in the "Your question" box.</li>
        <li><strong>Step 3:</strong> Click the "Predict Answer" button to get an answer based on the image.</li>
    </ul>
    <p style="color: #ff6600; font-weight: bold;">⚠️ Please note:</p>
    <p style="color: #333;">This app works best with simple questions related to the image. Complex questions might not be processed well because:</p>
    <ul style="color: #333;">
        <li>The model is optimized for answering direct and specific questions about objects in the image.</li>
        <li>It cannot describe the image in detail or provide complex explanations about the context.</li>
        <li>Complex questions that require detailed analysis or explanation might not produce accurate or meaningful answers.</li>
        <li>It’s best to ask questions that are clear and straightforward, focused on identifying or confirming visible objects in the image.</li>
    </ul>
    <p style="color: #333;">For example:</p>
    <ul style="color: #333;">
        <li><strong>Good Question:</strong> "Is the <object> present in the image?"</li>
        <li><strong>Not Ideal Question:</strong> "Can you describe the image in detail or explain the context of the objects?"</li>
    </ul>
    <p style="color: #333;">For best results, try to keep your questions clear and simple!</p>
    </div>
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
            # Get the answer
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
