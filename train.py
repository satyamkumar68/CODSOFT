# train.py

import os
import pickle
import numpy as np
import tensorflow as tf
import gc

# Memory optimizations
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

# Limit memory usage
tf.config.set_soft_device_placement(True)
tf.config.threading.set_intra_op_parallelism_threads(2)
tf.config.threading.set_inter_op_parallelism_threads(2)
from tensorflow import keras
from model import create_caption_decoder 
from data_prep import clean_caption, load_raw_caption_pairs
from config import (
    PATH_TOKENIZER,
    VOCAB_SIZE,
    MAX_CAPTION_LENGTH,
    BATCH_SIZE,
    EPOCHS,
    PATH_MODEL_WEIGHTS,
    PATH_FEATURES_TRAIN,
    PATH_TRAIN_IMAGES,
    FEATURE_DIM # Now 100352
)

# Define the sequence length for the generator input arrays (49)
SEQUENCE_LENGTH = MAX_CAPTION_LENGTH - 1 

# -----------------------------------------------------------
# --- 1. DATA GENERATOR ---
# -----------------------------------------------------------

def sequence_data_generator(all_features, caption_pairs, tokenizer, batch_size):
    """
    Creates a tf.data.Dataset for training the image captioning model.
    """
    # Filter pairs to only include those for which features were successfully extracted
    filtered_pairs = [p for p in caption_pairs if p['image_path'] in all_features]
    n_samples = len(filtered_pairs)
    
    if n_samples == 0:
        print("ERROR: No matching features found in the .npy file. Generator failed.")
        return None

    print(f"Generator created for {n_samples} filtered samples.")
    
    def gen_fn():
        while True:  # Loop indefinitely for Keras fit
            # Process one sample at a time
            idx = np.random.randint(0, n_samples)
            pair = filtered_pairs[idx]
            
            # Get image features
            image_features = all_features[pair['image_path']]
            
            # Process caption
            raw_caption = clean_caption(pair['raw_caption'])
            seq = tokenizer.texts_to_sequences([raw_caption])[0]
            padded_seq = keras.preprocessing.sequence.pad_sequences([seq], maxlen=MAX_CAPTION_LENGTH, padding='post')[0]
            
            # Split into input and target sequences
            caption_input = padded_seq[:-1]  # All but the last word
            caption_target = padded_seq[1:]   # All but the first word
            
            # Yield the training sample
            yield (
                {
                    'image_input': image_features,
                    'caption_input': caption_input
                },
                caption_target
            )
    
    # Define the shape and type of the data
    # Define the shape and type of the data
    output_signature = (
        {
            'image_input': tf.TensorSpec(shape=(FEATURE_DIM,), dtype=tf.float32),
            'caption_input': tf.TensorSpec(shape=(SEQUENCE_LENGTH,), dtype=tf.int32)
        },
        tf.TensorSpec(shape=(SEQUENCE_LENGTH,), dtype=tf.int32)
    )
    
    # Create and configure the dataset
    dataset = tf.data.Dataset.from_generator(
        gen_fn,
        output_signature=output_signature
    )
    
    return dataset

# -----------------------------------------------------------
# --- MAIN TRAINING EXECUTION ---
# -----------------------------------------------------------

def run_training():
    """Main training function."""
    print("--- Initializing Training Pipeline ---")

    # 1. Load the saved features
    try:
        features_path = 'train_image_features_subset.npy'
        with open(features_path, 'rb') as f:
            all_features = np.load(f, allow_pickle=True).item()
    except:
        print(f"ERROR: Could not load features from {PATH_FEATURES_TRAIN}")
        return
        
    # 2. Load the tokenizer and get caption pairs
    try:
        with open(PATH_TOKENIZER, 'rb') as f:
            tokenizer = pickle.load(f)
        filtered_pairs = load_raw_caption_pairs(PATH_TRAIN_IMAGES)
    except:
        print(f"ERROR: Could not load tokenizer from {PATH_TOKENIZER}")
        return

    # Create and train the model
    model = create_caption_decoder(tokenizer.word_index)
    print("\n--- Model Summary ---")
    model.summary()
    print("---------------------")

    # Start training
    print(f"\nStarting training for {EPOCHS} epochs on {len(filtered_pairs)} samples...")
    
    dataset = sequence_data_generator(all_features, filtered_pairs, tokenizer, BATCH_SIZE)
    dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    
    history = model.fit(
        dataset,
        steps_per_epoch=len(filtered_pairs) // BATCH_SIZE,
        epochs=EPOCHS,
        verbose=1
    )
    
    # Save the model
    model.save(PATH_MODEL_WEIGHTS)
    print(f"\nTraining complete! Model saved to {PATH_MODEL_WEIGHTS}")

if __name__ == '__main__':
    run_training()