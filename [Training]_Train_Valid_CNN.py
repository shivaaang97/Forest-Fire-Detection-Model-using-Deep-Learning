import os
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from PIL import Image
import json

# Setting the paths for the training and testing data
base_dir = 'C:\\forest_fire_data'
train_dir = os.path.join(base_dir, 'train')
test_dir = os.path.join(base_dir, 'test')

# Defining the data generators (minor preprocessing)
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest',
    validation_split=0.2 #Assigning 20% of the training data as validation data as the dataset has just the train and test data
)

test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(150, 150),
    batch_size=32,
    class_mode='categorical',
    subset='training'
)

validation_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(150, 150),
    batch_size=32,
    class_mode='categorical',
    subset='validation'
)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(150, 150),
    batch_size=32,
    class_mode='categorical',
    shuffle=False
)

# Preprocessing - Model architecture along with manual hyperparameter tuning 
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(512, activation='relu'),
    Dropout(0.5),
    Dense(3, activation='softmax')
])

# Compiling and training the model for interations
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
history = model.fit(
    train_generator,
    steps_per_epoch=100,
    epochs=7,
    validation_data=validation_generator,
    validation_steps=50
)

# Evaluating and saving the model
test_loss, test_acc = model.evaluate(test_generator, steps=50)
print(f"Test accuracy: {test_acc}")
model.save('C:/forest_fire_data/forestfire_detection_model.h5')
print("Model successfully saved!")

# Plotting the training and validation accuracy
import matplotlib.pyplot as plt
plt.figure()
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# Loading the trained model
loaded_model = load_model('forestfire_detection_model.h5')

# Function for predicting the category of an image
def predict_image_category(model, img_path):
    # Loading and preprocessing the image for prediction
    img = Image.open(img_path)
    img = img.resize((150, 150))
    img_array = np.array(img)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    # Make a prediction
    prediction = model.predict(img_array)
    predicted_class_index = np.argmax(prediction, axis=1)
    
    # Translate indices to class names
    classes = train_generator.class_indices
    classes = dict((v, k) for k, v in classes.items())
    return classes[predicted_class_index[0]]

# Defining the path to the image to test
img_path = 'C:\\forest_fire_data\\test\\smoke\\Smoke (3).jpg'

# Using this function to predict the category of the image
predicted_category = predict_image_category(loaded_model, img_path)
print("Predicted category:", predicted_category)

# Saving class_indices to JSON function
class_indices = train_generator.class_indices
with open('class_indices.json', 'w') as json_file:
    json.dump(class_indices, json_file)

print("Model and class indices have been saved!")

# Just for our reference purpose while running the prediction only script
class_indices = train_generator.class_indices
print("Class Indices:", class_indices)  # Display the class indices for reference
index_to_class = {v: k for k, v in class_indices.items()}  # Reverse mapping for predictions

