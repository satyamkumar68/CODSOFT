# data_prep.py

import json
import pickle
import os
import string
import tensorflow as tf
from tensorflow import keras
from config import PATH_ANNOTATIONS, VOCAB_SIZE, MAX_CAPTION_LENGTH, PATH_TOKENIZER, BASE_DATA_DIR

# --- Helper Function used in train.py and inference.py ---
def clean_caption(caption):
    """Adds start/end tokens, converts to lowercase, and removes punctuation."""
    caption = str(caption).lower()
    
    # Filter common punctuation 
    table = str.maketrans('', '', string.punctuation)
    caption = caption.translate(table)

    # Add start and end tokens
    caption = f"<start> {caption} <end>"
    return caption

# --- Function used by train.py to load pairs ---
def load_raw_caption_pairs(path_train_images):
    """
    Loads all captions, cleans, and returns the list of raw pairs aligned to image paths.
    """
    try:
        with open(PATH_ANNOTATIONS, 'r') as f:
            annotations = json.load(f)
    except FileNotFoundError:
        print(f"ERROR: Annotation file not found at {PATH_ANNOTATIONS}.")
        return []

    all_caption_pairs = []
    
    for ann in annotations['annotations']:
        image_id = ann['image_id']
        
        # The image path MUST match the key used in the .npy feature file
        image_path = os.path.join(
            path_train_images, 
            f"{image_id:012d}.jpg"
        )
        
        all_caption_pairs.append({
            'image_path': image_path,
            'raw_caption': ann['caption'] 
        })
        
    return all_caption_pairs
    
# --- Main Tokenizer Creation Function ---
def load_and_create_tokenizer():
    """
    Loads captions, creates vocabulary, and saves the tokenizer object.
    """
    print("Starting caption loading and tokenization...")
    
    try:
        with open(PATH_ANNOTATIONS, 'r') as f:
            annotations = json.load(f)
    except FileNotFoundError:
        print(f"ERROR: Annotation file not found at {PATH_ANNOTATIONS}. Please check paths.")
        return None, None
    except json.JSONDecodeError:
        print("ERROR: Annotation file is corrupted or empty. Check the JSON file integrity.")
        return None, None

    all_raw_captions = []
    
    for ann in annotations['annotations']:
        caption = clean_caption(ann['caption'])
        all_raw_captions.append(caption)
        
    # 1. Initialize and Fit Tokenizer
    tokenizer = keras.preprocessing.text.Tokenizer(
        num_words=VOCAB_SIZE,
        oov_token="<unk>",
        filters='!"#$%&()*+,-./:;=?@[\\]^_`{|}~ '
    )
    tokenizer.fit_on_texts(all_raw_captions)
    
    # Add padding token to index 0 (critical for Keras masking)
    tokenizer.word_index['<pad>'] = 0
    tokenizer.index_word[0] = '<pad>'
    
    print(f"Tokenizer created. Actual Vocabulary size: {len(tokenizer.word_index)}")
    
    # 2. Save the Tokenizer
    with open(PATH_TOKENIZER, 'wb') as f:
        pickle.dump(tokenizer, f)
    print(f"Tokenizer saved to {PATH_TOKENIZER}")
    
    return tokenizer, []
    

if __name__ == '__main__':
    load_and_create_tokenizer()