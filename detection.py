from tensorflow.keras.models import load_model

# Load the previously saved model
model = load_model('graffiti_detection_model.h5')

import numpy as np
from PIL import Image

def preprocess_image(image_path, target_size=(128, 128)):
    """
    Preprocess the input image for prediction by resizing, normalizing, and converting to numpy array.
    """
    img = Image.open(image_path)
    img = img.resize(target_size)  # Resize image to match the input size of the model
    img = np.array(img) / 255.0    # Normalize the image to [0, 1] range
    img = np.expand_dims(img, axis=0)  # Add batch dimension (1, 128, 128, 3)
    return img

def predict_graffiti(image_path):
    """
    Predict whether the input image has graffiti or not.
    """
    # Preprocess the image
    processed_image = preprocess_image(image_path)

    # Use the loaded model to make a prediction
    prediction = model.predict(processed_image)

    # Interpretation: if the prediction is > 0.5, it means graffiti is detected
    if prediction > 0.5:
        print("Graffiti detected!")
    else:
        print("No graffiti detected.")

# Example usage
image_path = 'example2.jpg'
predict_graffiti(image_path)