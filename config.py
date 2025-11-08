# config.py

import os

# --- 1. DATA PATHS ---

BASE_DATA_DIR = 'coco_dataset'

# Number of images to use for training (limited dataset)
MAX_IMAGES = 1000  # Using 1000 images for faster testing

# Correct paths for your nested folders
PATH_TRAIN_IMAGES = os.path.join(BASE_DATA_DIR, 'train2017', 'train2017')
PATH_ANNOTATIONS  = os.path.join(BASE_DATA_DIR, 'annotations_trainval2017', 'annotations', 'captions_train2017.json')
PATH_VAL_IMAGES   = os.path.join(BASE_DATA_DIR, 'val2017', 'val2017') 

# Paths for generated intermediate files
PATH_FEATURES_TRAIN = 'train_image_features.npy'
PATH_FEATURES_VAL   = 'val_image_features.npy'

PATH_TOKENIZER = 'tokenizer.pkl' 
PATH_MODEL_WEIGHTS = 'best_caption_model.h5'


# --- 2. MODEL HYPERPARAMETERS ---

# Feature dimension from ResNet50's average pooling layer
FEATURE_DIM = 2048  # ResNet50's standard output dimension
IMG_SIZE = (224, 224) 

# NLP Decoder (LSTM) Settings
VOCAB_SIZE = 5000       
MAX_CAPTION_LENGTH = 50 
EMBEDDING_DIM = 256     
UNITS = 512             


# --- 3. TRAINING PARAMETERS ---

BATCH_SIZE = 32
EPOCHS = 5  # Reduced from 20 to 5 for faster training
LEARNING_RATE = 1e-4 # 0.0001
VALIDATION_SPLIT = 0.1
BEAM_WIDTH = 3