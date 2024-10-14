import os
from PIL import Image
import numpy as np
from sklearn.model_selection import train_test_split

# Function to ensure all images are 3-channel (RGB)
def ensure_rgb(img):
    if img.mode != 'RGB':
        img = img.convert('RGB')
    return img

def load_images_from_directories(base_path, target_size=(128, 128)):
    images = []
    labels = []
    
    # Path for graffiti images
    graffiti_dir = os.path.join(base_path, 'graffiti_train')
    no_graffiti_dir = os.path.join(base_path, 'no_graffiti')
    
    # Load graffiti images and assign label 1
    for filename in os.listdir(graffiti_dir):
        img_path = os.path.join(graffiti_dir, filename)
        try:
            img = Image.open(img_path)
            img = ensure_rgb(img)  # Ensure 3-channel image
            img = img.resize(target_size)  # Resize to target size
            img = np.array(img) / 255.0  # Normalize pixel values to [0, 1]
            images.append(img)
            labels.append(1)  # Graffiti label
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
    
    # Load no_graffiti images and assign label 0
    for filename in os.listdir(no_graffiti_dir):
        img_path = os.path.join(no_graffiti_dir, filename)
        try:
            img = Image.open(img_path)
            img = ensure_rgb(img)  # Ensure 3-channel image
            img = img.resize(target_size)  # Resize to target size
            img = np.array(img) / 255.0  # Normalize pixel values to [0, 1]
            images.append(img)
            labels.append(1)  # Graffiti label
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
    
    return np.array(images), np.array(labels)

# Base directory containing the 'graffiti' and 'no_graffiti' folders
base_directory = ''

# Load images and labels
X, y = load_images_from_directories(base_directory)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Training samples: {X_train.shape[0]}, Testing samples: {X_test.shape[0]}")

import tensorflow as tf
from tensorflow.keras import layers, models

def create_model():
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Flatten())
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Create and train the model
model = create_model()
history = model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))

# Evaluate the model
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f'Test accuracy: {test_acc}')

# Plot accuracy and loss curves (optional for analysis)
import matplotlib.pyplot as plt

def plot_history(history):
    plt.plot(history.history['accuracy'], label='accuracy')
    plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.ylim([0, 1])
    plt.legend(loc='lower right')
    plt.show()

plot_history(history)

model.save('graffiti_detection_model.h5')