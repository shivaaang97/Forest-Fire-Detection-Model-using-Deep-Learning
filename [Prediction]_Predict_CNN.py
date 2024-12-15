# This script is for predicting any image of our choice by using our already trained model
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image

def load_and_prepare_image(img_path, target_size=(150, 150)):
    """Load an image file and prepare it for prediction."""
    img = Image.open(img_path)
    img = img.resize(target_size)
    img_array = np.array(img)
    img_array = img_array / 255.0  # Normalize to 0-1
    img_array = np.expand_dims(img_array, axis=0)  # Model expects a batch of images
    return img_array

def predict_image(model, img_array, class_indices):
    """Predict the category of an image using the loaded model and display category."""
    prediction = model.predict(img_array)
    predicted_class_index = np.argmax(prediction, axis=1)[0]
    class_names = {v: k for k, v in class_indices.items()}  # Reverse mapping of indices
    predicted_category = class_names[predicted_class_index]
    return predicted_class_index, predicted_category

# Loading the model
model_path = 'C:\\forest_fire_data\\forestfire_detection_model.h5'
model = load_model(model_path)

# Defining class indices to ensure it matches our training generator's indices
class_indices = {'fire': 0, 'no_fire': 1, 'smoke': 2}  # Update based on the dataset

# Path to the image we want to test
img_path = "C:\\forest_fire_data\\test\\no_fire\\02042.jpg"

# Loading and preparing the image
img_array = load_and_prepare_image(img_path)

# Making the prediction
predicted_class_index, predicted_category = predict_image(model, img_array, class_indices)
print("Predicted class index:", predicted_class_index)
print("Predicted category:", predicted_category)

