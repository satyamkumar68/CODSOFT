# feature_extract.py

import os
import pickle
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing import image
from tqdm import tqdm

# Import necessary paths and constants from your config file
from config import (
    PATH_TRAIN_IMAGES,
    PATH_FEATURES_TRAIN,
    IMG_SIZE,
    PATH_TOKENIZER
)

# --- MODEL DEFINITION (Copied from model.py for consistency) ---
def create_cnn_encoder():
    """Defines the pre-trained CNN model (ResNet50)."""
    image_model = ResNet50(
        include_top=False, 
        weights='imagenet', 
        input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3)
    )

    # Add a Global Average Pooling layer to get 2048-dim features
    hidden_layer = keras.layers.GlobalAveragePooling2D()(image_model.output)
    
    # Freeze the weights of the CNN 
    for layer in image_model.layers:
        layer.trainable = False

    cnn_encoder = Model(inputs=image_model.input, outputs=hidden_layer)
    return cnn_encoder

# -----------------------------------------------------------
# --- UTILITY FUNCTION ---
# -----------------------------------------------------------

def load_and_preprocess_image(image_path):
    """Loads and prepares image for the ResNet50 model."""
    try:
        img = image.load_img(image_path, target_size=IMG_SIZE)
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        return tf.keras.applications.resnet.preprocess_input(img_array)
    except Exception as e:
        return None

# -----------------------------------------------------------
# --- MAIN EXECUTION ---
# -----------------------------------------------------------

def main():
    # --- Read image paths from the directory ---
    try:
        image_filenames = [f for f in os.listdir(PATH_TRAIN_IMAGES) if f.endswith('.jpg')]
        unique_paths = [os.path.join(PATH_TRAIN_IMAGES, f) for f in image_filenames]
    except FileNotFoundError:
        print(f"ERROR: Image directory not found at {PATH_TRAIN_IMAGES}. Check your 'config.py' path.")
        return

    # 1. Initialize the Encoder Model
    cnn_encoder = create_cnn_encoder()
    
    # 2. Setup saving parameters
    OUTPUT_FEATURE_PATH = PATH_FEATURES_TRAIN.replace('.npy', '_subset.npy')
    
    # Limit to the first 1000 images for a quick test run
    images_to_process = unique_paths[:1000]
    
    # Dictionary to store feature vectors: {image_path: feature_vector}
    image_feature_mapping = {}

    print(f"Starting feature extraction for {len(images_to_process)} test images...")
    
    # 3. Feature Extraction Loop
    for path in tqdm(images_to_process, desc="Extracting Features"):
        processed_img = load_and_preprocess_image(path)
        
        if processed_img is not None:
            # Predict method returns the feature vector (now 2048)
            feature_vector = cnn_encoder.predict(processed_img, verbose=0)
            
            # Store the result: Key is the image path, value is the flattened feature vector
            # The saved feature must be the flat 2048 vector.
            image_feature_mapping[path] = feature_vector.flatten()
            
    # 4. Saving the Features
    np.save(OUTPUT_FEATURE_PATH, image_feature_mapping)
    print(f"\nFeature extraction complete. Saved {len(image_feature_mapping)} features to {OUTPUT_FEATURE_PATH}")

if __name__ == '__main__':
    main()