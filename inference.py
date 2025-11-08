# inference.py

import numpy as np
import tensorflow as tf
import os
from tensorflow.keras.models import Model
from data_prep import load_tokenizer, load_and_preprocess_image
from model import create_cnn_encoder, create_caption_decoder
from config import PATH_MODEL_WEIGHTS, IMG_SIZE, FEATURE_DIM, MAX_CAPTION_LENGTH, BEAM_WIDTH

# --- 1. Caption Generation Function ---

def beam_search_caption(image_features, tokenizer, model, beam_width, max_length):
    """Generates a caption for an image feature vector using Beam Search."""
    
    # Get ID of the start token
    start_token = tokenizer.word_index.get('<start>')
    if start_token is None:
        raise ValueError("<start> token not found in tokenizer vocabulary.")

    # beam: A list of tuples: (probability_score, [caption_sequence])
    # Initial beam has one sequence: the start token
    # We use log probability for stability in multiplication
    beam = [(0.0, [start_token])] 
    
    # Reshape image features to match the model's expected batch input (1, FEATURE_DIM)
    image_features = image_features[np.newaxis, :]

    for _ in range(max_length):
        new_beam = []
        
        for log_prob, seq in beam:
            # If the current sequence is complete, keep it and skip prediction
            if seq[-1] == tokenizer.word_index.get('<end>', -1): # Use -1 as fallback for safety
                new_beam.append((log_prob, seq))
                continue
            
            # Pad the current sequence to the model's input length
            input_sequence = tf.keras.preprocessing.sequence.pad_sequences(
                [seq], maxlen=max_length, padding='post', dtype='int32'
            )
            
            # Predict next word probabilities
            # The model expects [image_features (1, FEATURE_DIM), caption_sequence (1, MAX_CAPTION_LENGTH)]
            predictions = model.predict([image_features, input_sequence], verbose=0)[0]
            
            # Get the probability distribution for the *last* non-padded word
            # len(seq) - 1 is the index of the last word fed into the sequence
            last_word_predictions = predictions[len(seq) - 1]
            
            # Convert probabilities to log probabilities
            log_probs = np.log(last_word_predictions + 1e-10) # Add epsilon to avoid log(0)
            
            # Get the top N indices (N = beam_width)
            top_n_indices = np.argsort(log_probs)[-beam_width:]
            
            # Create new sequences
            for index in top_n_indices:
                word_log_prob = log_probs[index]
                # New probability = sum of log probabilities
                new_log_prob = log_prob + word_log_prob
                new_seq = seq + [index]
                new_beam.append((new_log_prob, new_seq))
        
        # Sort and select the top 'beam_width' sequences from the entire set
        # We sort by the highest log probability (which is closest to 0, e.g., -1 > -5)
        new_beam.sort(key=lambda x: x[0], reverse=True)
        beam = new_beam[:beam_width]
        
        # Stop condition: if the best sequence has terminated
        if beam[0][1][-1] == tokenizer.word_index.get('<end>', -1):
            break

    # 2. Final Output Selection and Conversion
    best_prob, best_sequence = beam[0]
    
    # Convert sequence of IDs back to words
    final_caption = [tokenizer.index_word[i] for i in best_sequence if i > 0]
    
    # Clean up the <start> and <end> tokens for final output
    caption_string = ' '.join(final_caption)
    caption_string = caption_string.replace("<start>", "").replace("<end>", "").strip()
    
    return caption_string

# --- 2. Main Execution (Inference) ---

def run_inference(image_path):
    """Loads model, extracts features, and generates caption for a single image."""
    print(f"\n--- Running Inference for: {image_path} ---")

    # 1. Load Pre-Trained Models/Data
    tokenizer = load_tokenizer()
    if not tokenizer:
        return "Could not load tokenizer. Please run data_prep.py first."
    
    # Initialize the CNN Encoder (needed to extract features from the new image)
    cnn_encoder = create_cnn_encoder()
    
    # Initialize the Decoder Model
    vocab_size = len(tokenizer.word_index)
    caption_model = create_caption_decoder(vocab_size)
    
    # Load the trained weights
    try:
        caption_model.load_weights(PATH_MODEL_WEIGHTS)
    except FileNotFoundError:
        return f"Could not load model weights from {PATH_MODEL_WEIGHTS}. Please run train.py first."

    # 2. Extract Feature Vector from New Image
    try:
        processed_img = load_and_preprocess_image(image_path, IMG_SIZE)
        image_feature_vector = cnn_encoder.predict(processed_img, verbose=0).flatten()
    except Exception as e:
        return f"Error extracting features from image: {e}"

    # 3. Generate Caption using Beam Search
    caption = beam_search_caption(
        image_feature_vector,
        tokenizer,
        caption_model,
        BEAM_WIDTH,
        MAX_CAPTION_LENGTH
    )
    
    return f"Generated Caption (Beam Search, width={BEAM_WIDTH}): {caption}"

if __name__ == '__main__':
    # --- EXAMPLE USAGE ---
    # NOTE: You must provide a valid image path from your extracted dataset here
    # Example path assuming a COCO image ID 397133 is in the val2017 folder
    # Replace this with an actual image path after extraction!
    
    example_img_path = os.path.join("coco_dataset", "val2017", "000000397133.jpg")
    
    # You must run data_prep.py, feature_extract.py, and train.py before this will work!
    # For now, it will likely return an error saying the feature/weight file is missing.
    
    print(run_inference(example_img_path))