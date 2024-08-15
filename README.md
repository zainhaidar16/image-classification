# 🖼️ Image Classification App

A Streamlit app for uploading an image and comparing predictions from multiple pre-trained image classification models. The app uses popular models like ResNet-50, Inception v3, VGG-16, DenseNet-121, and MobileNet v2.

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://image-classifications.streamlit.app/)

### Features

- **Image Upload**: Upload images in JPG, JPEG, or PNG formats directly through the app interface.
- **Model Comparison**: Choose from five pre-trained models for image classification:
  - **ResNet-50**: A deep residual network with 50 layers.
  - **Inception v3**: A deep convolutional network with multiple inception modules.
  - **VGG-16**: A classic deep learning model with 16 layers.
  - **DenseNet-121**: A densely connected convolutional network with 121 layers.
  - **MobileNet v2**: A lightweight model designed for mobile and embedded vision applications.
- **Customizable Settings**:
  - **Model Selection**: Select which models to include in the classification process.
  - **Confidence Threshold**: Use a slider to filter predictions based on confidence score. Only predictions above this threshold are shown.
  - **Top N Predictions**: Adjust the number of top predictions displayed for each model. 
- **Interactive Results**: View and compare predictions from all selected models in a table format.
- **Personal Information**: A footer with links to the developer’s GitHub, LinkedIn, and portfolio for easy access.

### Code Overview

- **`streamlit_app.py`**: This is the main script for the Streamlit application. It contains the core logic for:
  - Loading pre-trained models from `torchvision`.
  - Handling image uploads and preprocessing.
  - Running image classification through selected models.
  - Displaying results and interactive widgets in the Streamlit interface.

- **`requirements.txt`**: This file lists all the Python packages required to run the application. It includes dependencies such as:
  - `streamlit` for the web app framework.
  - `torch` and `torchvision` for model loading and image transformations.
  - `Pillow` for image processing.
  - `pandas` for data handling and displaying results in tables.

- **`imagenet-simple-labels.json`**: A JSON file containing the class labels used by the models to map prediction indices to human-readable labels. This file is necessary for interpreting the output of the models.

### How to run it on your own machine

1. Install the requirements

   ```
   $ pip install -r requirements.txt
   ```

2. Run the app

   ```
   $ streamlit run streamlit_app.py
   ```

### Customizing the App

You can tailor the app to fit specific needs or preferences by modifying the following aspects:

- **Model Selection**:
  - In the sidebar, you can choose which pre-trained models to use for image classification. By default, the app includes ResNet-50, Inception v3, VGG-16, DenseNet-121, and MobileNet v2. To add or remove models, update the model selection logic in the `streamlit_app.py` file.

- **Confidence Threshold**:
  - Adjust the confidence threshold using the slider in the sidebar. Predictions with a confidence score below this threshold will be filtered out. You can change the range and default value of the threshold slider by modifying the relevant code in `streamlit_app.py`.

- **Top N Predictions**:
  - Use the slider in the sidebar to set the number of top predictions displayed for each model. Modify the slider settings to adjust the range and default value by editing the `streamlit_app.py` file.

- **Personal Information**:
  - Update the footer to include your personal information and links. Change the URLs and text in the footer section of `streamlit_app.py` to reflect your GitHub, LinkedIn, and portfolio links.

- **ImageNet Labels**:
  - If you need to update or modify the class labels used by the models, edit the `imagenet-simple-labels.json` file or provide a new labels file in JSON format. Ensure that the labels correspond correctly to the models' output classes.

- **Styling and Layout**:
  - Customize the appearance of the app by modifying Streamlit components and layout options in `streamlit_app.py`. You can adjust the layout, colors, and styling to match your preferences.

By making these adjustments, you can tailor the app to better suit your requirements or preferences.

### Troubleshooting

If you encounter issues while running the app, try the following solutions:

- **403 Errors**:
  - **Issue**: You might encounter a 403 Forbidden error if the app fails to access resources or download files.
  - **Solution**: Ensure that the URLs used for downloading files or accessing external resources are correct and accessible. Verify that you have permission to access these resources. If you're behind a firewall or proxy, make sure it’s configured properly.

- **AttributeError**:
  - **Issue**: Errors like `AttributeError: 'Tensor' object has no attribute 'logits'` indicate compatibility issues between the PyTorch version and the model code.
  - **Solution**: Ensure that your versions of `torch` and `torchvision` are compatible with the code. You might need to upgrade or downgrade these packages. Check the versions specified in the `requirements.txt` file and ensure they match the versions used in the code.

- **Model Download Issues**:
  - **Issue**: Problems with downloading pre-trained model weights might occur due to network issues or incorrect URLs.
  - **Solution**: Verify that your internet connection is stable. Check the URLs for downloading models and make sure they are correct. You can manually download the models if necessary and place them in the correct directory.

- **Dependency Conflicts**:
  - **Issue**: Conflicts between package versions can lead to runtime errors or unexpected behavior.
  - **Solution**: Ensure all dependencies are installed as specified in the `requirements.txt` file. If conflicts arise, consider creating a new virtual environment and reinstalling the dependencies. You can also use tools like `pipdeptree` to identify and resolve dependency issues.

- **File Not Found Errors**:
  - **Issue**: Errors related to missing files (e.g., `imagenet-simple-labels.json`) can occur if the required files are not in the project directory.
  - **Solution**: Ensure all necessary files are present in the project directory. Follow the instructions in the "Download ImageNet Labels" section to place the `imagenet-simple-labels.json` file in the correct location.

- **General Debugging**:
  - **Issue**: Other unexpected issues or errors.
  - **Solution**: Check the error messages for clues about what went wrong. Review the code and logs for more information. Consult the documentation for Streamlit, PyTorch, and other libraries used in the app for additional troubleshooting tips.

If problems persist, consider reaching out to the community or opening an issue on the project's GitHub repository for further assistance.
