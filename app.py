from flask import Flask, render_template, request, redirect, url_for
import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
from tensorflow.keras.models import load_model
import pickle

# Initialize the Flask app
app = Flask(__name__)

# Path to upload folder
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Step 1: Load the trained model
model = load_model('Project 3/model/ai_lungs_disease.h5')  # Adjust the path to your model file

# Step 2: Load the LabelEncoder
with open('Project 3/model/Label_encoder (2).pkl', 'rb') as file:
    label_encoder = pickle.load(file)

# Step 3: Define the detection function
def detection_system(image_path, model, label_encoder, image_size=(400, 400)):
    """
    Detection system function to classify an input image.

    Parameters:
    - image_path: The path to the image to classify.
    - model: The trained Keras model.
    - label_encoder: The LabelEncoder used to encode the labels.
    - image_size: The target size for resizing the image (default: 400x400).

    Returns:
    - predicted_label: The predicted class label.
    - confidence_score: The confidence score of the predicted class.
    """
    # Load the image from the provided path
    image = cv2.imread(image_path)

    if image is None:
        raise ValueError(f"Image not found at path: {image_path}")

    # Convert image from BGR to RGB format
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Resize the image to the same size as the training images
    image_resized = cv2.resize(image_rgb, image_size)

    # Normalize the pixel values to the range [0, 1]
    image_normalized = image_resized / 255.0

    # Expand the dimensions to match the input shape for the model
    image_input = np.expand_dims(image_normalized, axis=0)

    # Predict the class of the image
    predictions = model.predict(image_input)

    # Get the predicted class index and confidence score
    predicted_index = np.argmax(predictions)
    confidence_score = predictions[0][predicted_index]

    # Decode the predicted index back to the original label
    predicted_label = label_encoder.inverse_transform([predicted_index])[0]

    return predicted_label, confidence_score

# Step 4: Create routes for the Flask app

@app.route('/')
def index():
    return render_template('index.html')  # Render an HTML form for file upload

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return redirect(request.url)
    
    file = request.files['file']

    if file.filename == '':
        return redirect(request.url)

    if file:
        # Save the uploaded image to the upload folder
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(file_path)

        # Run the detection system on the uploaded image
        predicted_label, confidence_score = detection_system(file_path, model, label_encoder)

        # Return the result to the user
        return render_template('result.html', label=predicted_label, confidence=confidence_score*100)

# Step 5: Run the app
if __name__ == '__main__':
    app.run(debug=True)
