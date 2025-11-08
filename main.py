# main.py

import os
from data_prep import load_and_create_tokenizer
from feature_extract import extract_and_save_features
from train import run_training
from inference import run_inference
from config import PATH_TRAIN_IMAGES

def run_project_pipeline():
    """Runs the full sequence: Data Prep -> Feature Extraction -> Training -> Example Inference."""
    print("--- Starting Image Captioning Project Pipeline ---")

    # STEP 1: Data Preparation and Tokenizer
    # This also saves the tokenizer.pkl file
    print("\n[STEP 1] Running Data Preparation...")
    tokenizer, training_pairs = load_and_create_tokenizer()
    
    if not tokenizer or not training_pairs:
        print("Pipeline aborted due to missing data/tokenizer.")
        return

    # STEP 2: Feature Extraction
    # This requires the images to be fully downloaded and extracted!
    print("\n[STEP 2] Running Feature Extraction (Requires images in 'coco_dataset/train2017')...")
    # Note: feature_extract.py has its own main block to run the extraction process.
    # For this high-level pipeline, you'll need to run feature_extract.py separately
    # to avoid re-running the long extraction process every time.
    
    # For now, we will assume feature_extract.py has been run and features are saved.
    
    # STEP 3: Training
    # This requires the feature file (train_image_features.npy) to exist!
    print("\n[STEP 3] Running Training...")
    # run_training() # Uncomment this once you are ready to train!
    print("Training module skipped. Please run 'python train.py' separately to start training.")

    # STEP 4: Inference Example
    print("\n[STEP 4] Running Inference Example...")
    # NOTE: Change this path to a real image from your VAL or TEST set after extraction.
    example_img_path = os.path.join("coco_dataset", "val2017", "000000397133.jpg")
    
    # Check if the example image path exists before running inference
    if os.path.exists(example_img_path):
        # result = run_inference(example_img_path) # Uncomment this once model is trained!
        # print(result)
        print("Inference skipped. Please train the model and update the example image path.")
    else:
        print(f"Inference skipped. Example image not found at: {example_img_path}")

if __name__ == '__main__':
    # NOTE: You will need to manually run the following scripts in order:
    # 1. python data_prep.py
    # 2. python feature_extract.py (This is the long one)
    # 3. python train.py
    # 4. python inference.py (To test the final model)
    
    print("Project files generated. Please run them sequentially in your terminal as the data becomes available.")