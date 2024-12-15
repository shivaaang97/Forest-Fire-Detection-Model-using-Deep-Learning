from flask import Flask, request, render_template_string, jsonify
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import os
from werkzeug.utils import secure_filename
import logging

# Initializing Flask app and set up logging
app = Flask(__name__)
logging.basicConfig(level=logging.INFO)

# Loading the already trained model 
MODEL_PATH = 'C:/forest_fire_data/forestfire_detection_model.h5'
model = load_model(MODEL_PATH)

# Simple HTML template for the interface
HTML_TEMPLATE = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Forest Fire Detection</title>
</head>
<body>
    <h1>Forest Fire Image Classification</h1>
    <form method="post" action="/predict" enctype="multipart/form-data">
        <input type="file" name="file" required>
        <input type="submit" value="Upload and predict">
    </form>
    {% if prediction %}
        <h2>Prediction: {{ prediction }}</h2>
    {% endif %}
</body>
</html>
'''

@app.route('/', methods=['GET'])
def index():
    # Serving the main interface
    return render_template_string(HTML_TEMPLATE)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Checking if the post request has the file part
        if 'file' not in request.files:
            return jsonify({'error': 'No file part in the request'}), 400
        
        file = request.files['file']
        
        if file.filename == '':
            return jsonify({'error': 'No file selected for uploading'}), 400
        
        if file:
            # Ensuring that the directory exists
            upload_folder = os.path.join('c:/forest_fire_data', 'uploads')
            os.makedirs(upload_folder, exist_ok=True)
            
            # Securing the filename and saving the file
            filename = secure_filename(file.filename)
            file_path = os.path.join(upload_folder, filename)
            file.save(file_path)
            logging.info(f"File saved to {file_path}")
            
            # Processing the file for prediction
            img = Image.open(file_path)
            img = img.resize((150, 150))
            img_array = np.array(img) / 255.0
            img_array = np.expand_dims(img_array, axis=0)
            prediction = model.predict(img_array)
            predicted_class_index = np.argmax(prediction, axis=1)
            
            # Mapping index to class
            classes = {0: 'Fire', 1: 'No Fire', 2: 'Smoke'}
            predicted_class = classes.get(predicted_class_index[0], 'Unknown')
            
            return render_template_string(HTML_TEMPLATE, prediction=predicted_class)
    except Exception as e:
        logging.error(f"An error occurred: {str(e)}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, use_reloader=False)
