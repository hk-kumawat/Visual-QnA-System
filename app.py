import streamlit as st
from PIL import Image
from io import BytesIO
from transformers import ViltProcessor, ViltForQuestionAnswering, BlipProcessor, BlipForConditionalGeneration, GPT2LMHeadModel, GPT2Tokenizer

# Set page layout to wide
st.set_page_config(page_title="Visual Question Answering", layout="wide")

# Load VQA model
processor = ViltProcessor.from_pretrained("dandelin/vilt-b32-finetuned-vqa")
model = ViltForQuestionAnswering.from_pretrained("dandelin/vilt-b32-finetuned-vqa")

# Load image captioning model (BLIP)
blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

# Load GPT model for text generation (elaboration)
gpt_tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
gpt_model = GPT2LMHeadModel.from_pretrained("gpt2")

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

# Function to elaborate on a short answer using GPT
def elaborate_answer(answer, caption):
    try:
        # Generate a prompt to ask GPT to elaborate on the answer
        prompt = f"The answer to the question is: '{answer}'. Based on the image caption: '{caption}', please provide a detailed description."
        
        # Encode the prompt
        inputs = gpt_tokenizer.encode(prompt, return_tensors="pt")

        # Generate response using GPT
        output = gpt_model.generate(inputs, max_length=100, num_return_sequences=1, no_repeat_ngram_size=2)
        
        # Decode the generated output
        elaborated_answer = gpt_tokenizer.decode(output[0], skip_special_tokens=True)
        
        return elaborated_answer

    except Exception as e:
        return str(e)

# Set up the Streamlit app
st.title("🔍 Visual Question Answering 🖼️ ")
st.write("Upload an image and choose or write a question to get an answer!")

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

        # Display the caption with previous formatting and centered
        st.markdown(f"<div class='centered'>{caption}</div>", unsafe_allow_html=True)

        # Initialize an empty list for suggested questions (can add later if required)
        suggested_questions = []
    else:
        suggested_questions = []  # Handle case where no image is uploaded

# Question Input (Write your own question)
with col2:
    # Store the selected question in session state to persist its value
    if 'question' not in st.session_state:
        st.session_state['question'] = ""

    # Allow user to type their own question
    question = st.text_input("Your question", value=st.session_state['question'])

    # Button for prediction (only show if custom question is selected)
    if uploaded_file and question:
        if st.button("Predict Answer"):
            image_bytes = uploaded_file.getvalue()

            # Get the short answer from VQA
            short_answer = get_answer(image_bytes, question)

            # Elaborate the answer using GPT
            elaborated_answer = elaborate_answer(short_answer, caption)

            # Display the elaborated answer
            st.success(f"Detailed Answer: {elaborated_answer}")

            # After showing the answer, reset question input field for new input
            st.session_state['question'] = ""  # Clear the 'Your question' box for next input

# Footer
st.markdown("---")
st.markdown(
    "<p style='text-align: center;'>Where <strong>Vision</strong> Meets <strong>Intelligence</strong> - A Creation by <strong>Harshal Kumawat</strong> 👁️🤖</p>",
    unsafe_allow_html=True
)
