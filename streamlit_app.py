import streamlit as st
from PIL import Image
import torch
import torchvision.transforms as transforms
from torchvision import models
import torch.nn.functional as F
import json

# Load ImageNet labels locally
def load_labels():
    try:
        with open("imagenet-simple-labels.json") as f:
            labels = json.load(f)
        st.write("Labels loaded successfully.")
    except Exception as e:
        st.error(f"Failed to load labels: {e}")
        labels = None
    return labels

# Function to load the selected model
def load_model(model_name):
    try:
        if model_name == "ResNet-50":
            model = models.resnet50(pretrained=True)
        elif model_name == "Inception v3":
            model = models.inception_v3(pretrained=True)
        elif model_name == "VGG-16":
            model = models.vgg16(pretrained=True)
        elif model_name == "DenseNet-121":
            model = models.densenet121(pretrained=True)
        elif model_name == "MobileNet v2":
            model = models.mobilenet_v2(pretrained=True)
        else:
            st.error("Unknown model selected!")
            return None
        model.eval()
        st.write("Model loaded successfully.")
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        model = None
    return model

# Streamlit app layout
st.set_page_config(page_title="Image Classification App", layout="centered")

# Header section
st.title("🖼️ Image Classification App")
st.write("Upload an image and the app will classify it using a pre-trained model.")

# Sidebar for additional settings
st.sidebar.header("Settings")

# Model selection
model_name = st.sidebar.selectbox(
    "Choose a pre-trained model:",
    ("ResNet-50", "Inception v3", "VGG-16", "DenseNet-121", "MobileNet v2")
)

# Confidence threshold slider
threshold = st.sidebar.slider(
    "Confidence Threshold",
    min_value=0.0,
    max_value=1.0,
    value=0.5,
    step=0.01,
    help="Filter predictions based on the confidence score. Only predictions above this threshold will be displayed."
)

# Load the selected model
model = load_model(model_name)

# Load labels
labels = load_labels()

# Define the image preprocessing steps
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Open the image file
    image = Image.open(uploaded_file)

    # Display the uploaded image
    st.image(image, caption="Uploaded Image", use_column_width=True)
    st.write("")

    # Preprocess the image
    img_tensor = preprocess(image)
    img_tensor = img_tensor.unsqueeze(0)  # Add batch dimension

    # Display a progress bar while the prediction is being made
    with st.spinner(f"Classifying using {model_name}..."):
        # Make prediction
        with torch.no_grad():
            output = model(img_tensor)
            # No need to access .logits, just use the output directly
            probabilities = F.softmax(output[0], dim=0)

        # Get the top 5 predictions
        top5_prob, top5_catid = torch.topk(probabilities, 5)

    # Display the results filtered by the threshold
    st.subheader("Prediction Results")
    for i in range(top5_prob.size(0)):
        if top5_prob[i].item() >= threshold:
            st.write(f"**{labels[top5_catid[i]]}**: {top5_prob[i].item()*100:.2f}%")
        else:
            st.write(f"*Confidence for {labels[top5_catid[i]]} is below the threshold*")

    # Add an expander to display more information
    with st.expander("See explanation"):
        st.write(f"""
            The model used for classification is **{model_name}**. 
            It is pre-trained on the ImageNet dataset, which contains over 1 million images across 1000 categories.
            The probabilities shown above are the model's confidence in each prediction.
            You can adjust the confidence threshold to filter out less certain predictions.
        """)

else:
    st.info("Please upload an image file to get started.")

# Footer section
st.markdown(
    """
    <style>
    footer {visibility: hidden;}
    .footer {
        position: fixed;
        bottom: 0;
        left: 0;
        width: 100%;
        background-color: #803DF5;
        text-align: center;
        padding: 10px;
    }
    </style>
    <div class="footer">
        <p>Developed with ❤️ by Zain Haidar</p>
    </div>
    """, unsafe_allow_html=True
)
