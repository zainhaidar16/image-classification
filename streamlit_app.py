import streamlit as st
from PIL import Image
import torch
import torchvision.transforms as transforms
from torchvision import models
import torch.nn.functional as F
import json

# Load ImageNet labels locally
def load_labels():
    with open("imagenet-simple-labels.json") as f:
        labels = json.load(f)
    return labels

# Function to load all models
def load_all_models():
    models_dict = {
        "ResNet-50": models.resnet50(weights=models.ResNet50_Weights.DEFAULT),
        "Inception v3": models.inception_v3(weights=models.Inception_V3_Weights.DEFAULT),
        "VGG-16": models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1),
        "DenseNet-121": models.densenet121(weights=models.DenseNet121_Weights.DEFAULT),
        "MobileNet v2": models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT),
    }
    for name, model in models_dict.items():
        model.eval()  # Set model to evaluation mode
    return models_dict

# Streamlit app layout
st.set_page_config(page_title="Image Classification App", layout="centered")

# Header section
st.title("🖼️ Image Classification App - Model Comparison")
st.write("Upload an image and compare predictions from multiple pre-trained models.")

# Sidebar for additional settings
st.sidebar.header("Settings")

# Model selection checkbox
st.sidebar.subheader("Select Models to Compare")
selected_models = st.sidebar.multiselect(
    "Choose the models:",
    ["ResNet-50", "Inception v3", "VGG-16", "DenseNet-121", "MobileNet v2"],
    default=["ResNet-50", "Inception v3", "VGG-16", "DenseNet-121", "MobileNet v2"]
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

# Option to display top N predictions
top_n = st.sidebar.slider(
    "Number of Top Predictions to Display",
    min_value=1,
    max_value=5,
    value=5,
    help="Specify the number of top predictions to display for each model."
)

# Load the models and labels
models_dict = load_all_models()
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

    # Initialize results list
    results = []

    # Loop through each selected model and make predictions
    for model_name in selected_models:
        model = models_dict[model_name]
        with st.spinner(f"Classifying using {model_name}..."):
            with torch.no_grad():
                output = model(img_tensor)
                probabilities = F.softmax(output[0], dim=0)

            # Get the top N predictions
            top_prob, top_catid = torch.topk(probabilities, top_n)

            # Store the results
            for i in range(top_prob.size(0)):
                if top_prob[i].item() >= threshold:
                    results.append((model_name, labels[top_catid[i]], top_prob[i].item()))

    # Convert results to a dataframe for display
    import pandas as pd
    df_results = pd.DataFrame(results, columns=["Model", "Predicted Class", "Confidence"])
    df_results["Confidence"] = df_results["Confidence"].apply(lambda x: f"{x*100:.2f}%")

    # Display the results in a table
    st.subheader("Model Comparison")
    st.dataframe(df_results)

else:
    st.info("Please upload an image file to get started.")

# Footer section with personal details and links
st.markdown(
    """
    <style>
    footer {visibility: hidden;}
    .footer {
        position: fixed;
        bottom: 0;
        left: 0;
        width: 100%;
        background-color: #f1f1f1;
        text-align: center;
        padding: 10px;
        font-size: 14px;
        color: #333;
    }
    </style>
    <div class="footer">
        <p>Developed with ❤️ by <a href="https://your-portfolio-link.com" target="_blank">Zain Haidar</a></p>
        <p>
            <a href="https://github.com/zainhadiar16" target="_blank">GitHub</a> |
            <a href="https://www.linkedin.com/in/your-linkedin/" target="_blank">LinkedIn</a> |
            <a href="https://zaintheanalyst.com" target="_blank">Portfolio</a>
        </p>
    </div>
    """, unsafe_allow_html=True
)
