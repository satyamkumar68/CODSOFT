# model.py

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Model
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import (
    Input, Dense, Embedding, LSTM, Dropout,
    RepeatVector, Concatenate, TimeDistributed
)
from tensorflow.keras.optimizers import Adam
from config import (
    FEATURE_DIM, 
    MAX_CAPTION_LENGTH, 
    EMBEDDING_DIM, 
    UNITS, 
    VOCAB_SIZE, 
    IMG_SIZE, 
    LEARNING_RATE
)

# Define the sequence length for input layers (MAX_CAPTION_LENGTH - 1 = 49)
SEQUENCE_LENGTH = MAX_CAPTION_LENGTH - 1

def create_cnn_encoder():
    """Defines the pre-trained CNN model (ResNet50)."""
    image_model = ResNet50(
        include_top=False, 
        weights='imagenet', 
        input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3)
    )

    hidden_layer = image_model.layers[-2].output 
    
    # Freeze the weights of the CNN 
    for layer in image_model.layers:
        layer.trainable = False

    cnn_encoder = Model(inputs=image_model.input, outputs=hidden_layer)
    return cnn_encoder

def create_caption_decoder(vocab_size):
    """
    Defines the full Encoder-Decoder Keras model, expecting inputs of length 49.
    """
    # --- 1. Image Feature Input (Input 1) ---
    # FIX: Input shape is 100352
    image_input = Input(shape=(FEATURE_DIM,), name='image_input')
    
    # FIX: Dense layer projects 100352 down to 512 units
    feature_projection = Dense(UNITS, activation='relu', name='feature_projection')(image_input)
    feature_projection = Dropout(0.5)(feature_projection)
    
    # Repeat the projected feature vector across the sequence length (49)
    image_context_repeated = RepeatVector(SEQUENCE_LENGTH)(feature_projection)
    
    # --- 2. Sequence Input (Input 2) ---
    # Input shape is correctly set to 49
    caption_input = Input(shape=(SEQUENCE_LENGTH,), name='caption_input')
    
    # Word Embedding Layer
    caption_embedding = Embedding(
        input_dim=VOCAB_SIZE + 1,  # Size of the vocabulary plus padding token
        output_dim=EMBEDDING_DIM,   
        mask_zero=True,
        name='caption_embedding'
    )(caption_input)

    # --- 3. Combine Inputs and LSTM Decoder ---
    
    # Concatenate the repeated image context (49 steps) with the word embeddings (49 steps)
    combined_input = Concatenate()([image_context_repeated, caption_embedding])
    
    # Recurrent Layer (LSTM)
    decoder_lstm = LSTM(
        UNITS, 
        return_sequences=True, 
        name='decoder_lstm'
    )(combined_input)
    
    decoder_dropout = Dropout(0.5)(decoder_lstm)

    # --- 4. Output Layer ---
    # TimeDistributed ensures the Dense prediction layer runs on every time step (word)
    decoder_output = TimeDistributed(
        Dense(VOCAB_SIZE + 1, activation='softmax'),  # Add 1 for padding token
        name='output_prediction'
    )(decoder_dropout)

    # 5. Full Model definition
    full_caption_model = Model(
        inputs=[image_input, caption_input], 
        outputs=decoder_output
    )

    # 6. Compile
    full_caption_model.compile(optimizer=Adam(learning_rate=LEARNING_RATE), loss='sparse_categorical_crossentropy')

    return full_caption_model

if __name__ == '__main__':
    model = create_caption_decoder(VOCAB_SIZE)
    model.summary()