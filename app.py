import streamlit as st
from PIL import Image
import requests
from io import BytesIO
import torch
from transformers import ViltProcessor, ViltForQuestionAnswering
from torchvision import models, transforms
import matplotlib.pyplot as plt
import numpy as np

# Set page layout to wide
st.set_page_config(layout="wide")

# Load the model for VQA (Visual Question Answering)
processor = ViltProcessor.from_pretrained("dandelin/vilt-b32-finetuned-vqa")
model = ViltForQuestionAnswering.from_pretrained("dandelin/vilt-b32-finetuned-vqa")

# Load pre-trained Faster R-CNN model for object detection
object_detection_model = models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
object_detection_model.eval()

# Preprocessing for object detection
transform = transforms.Compose([
    transforms.ToTensor(),
])

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

# Image analysis function using Faster R-CNN
def analyze_image(image):
    # Convert image to tensor and pass through the object detection model
    img_tensor = transform(image).unsqueeze(0)
    
    with torch.no_grad():
        prediction = object_detection_model(img_tensor)

    # Get the boxes, labels, and scores
    boxes = prediction[0]['boxes'].numpy()
    labels = prediction[0]['labels'].numpy()
    scores = prediction[0]['scores'].numpy()

    return boxes, labels, scores

def plot_image_with_boxes(image, boxes, labels, scores):
    # Convert image to numpy array for plotting
    image = np.array(image)

    # Create a plot to visualize the boxes and labels
    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    
    for box, label, score in zip(boxes, labels, scores):
        if score > 0.5:  # Filter out low confidence detections
            xmin, ymin, xmax, ymax = box
            plt.gca().add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, 
                                               linewidth=2, edgecolor="r", facecolor="none"))
            plt.text(xmin, ymin, f"Label: {label}, Score: {score:.2f}", 
                     fontsize=12, bbox=dict(facecolor="yellow", alpha=0.5))

    st.pyplot(plt)

# Set up the Streamlit app
st.title("Visual Question Answering & Image Analysis")
st.write("Upload an image, ask a question, and get answers along with image analysis.")

# Create columns for image upload and input fields
col1, col2 = st.columns(2)

# Image upload
with col1:
    uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        st.image(uploaded_file, use_container_width=True)

# Question input
with col2:
    question = st.text_input("Question")

    # Process the image and question when both are provided
    if uploaded_file and question:
        image = Image.open(uploaded_file)
        image_byte_array = BytesIO()
        image.save(image_byte_array, format='JPEG')
        image_bytes = image_byte_array.getvalue()

        # Get the answer
        answer = get_answer(image_bytes, question)

        # Display the answer
        st.success("Answer: " + answer)

# Image analysis (Object detection)
if uploaded_file:
    image = Image.open(uploaded_file)
    boxes, labels, scores = analyze_image(image)
    st.subheader("Image Analysis: Detected Objects")
    plot_image_with_boxes(image, boxes, labels, scores)
