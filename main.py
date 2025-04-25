#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Guitar Chord Recognition using GuitarSet
Main entry point for the subproject

Acknowledgment: This project has been enhanced with the assistance of Claude AI by Anthropic
Claude helped with code improvements, debugging, and implementation of the audio segmentation approach.
"""

import os
import argparse
import shutil
from pathlib import Path
import sys
import torch  # Ensure torch is imported here

# Try to import torch again in case there was an issue
try:
    import torch
except ImportError:
    print("Error: PyTorch is not installed. Please install it with 'pip install torch'.")
    sys.exit(1)

# Add the project directory to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from guitarset_chord_recognition.utils.preprocessing import process_dataset, get_data_splits
from guitarset_chord_recognition.utils.data_loader import create_data_loaders
from guitarset_chord_recognition.model_comparison import train_models, evaluate_models, compare_models, setup_directories


# Default paths
DEFAULT_DATA_DIR = "/home/masa/adv_ml/GuitarSet"
DEFAULT_PROCESSED_DIR = "./guitarset_chord_recognition/data/processed"
DEFAULT_MODELS_DIR = "./guitarset_chord_recognition/models"
DEFAULT_RESULTS_DIR = "./guitarset_chord_recognition/results"

def copy_raw_data(source_dir, destination_dir, subdirs=None):
    """
    Copy necessary raw data from source to destination directory
    
    Args:
        source_dir: Source directory in ADV_ML
        destination_dir: Destination directory in project
        subdirs: List of subdirectories to search for audio files (if None, search root)
    """
    # Create destination directory if it doesn't exist
    os.makedirs(destination_dir, exist_ok=True)
    
    print(f"Copying data from {source_dir} to {destination_dir}...")
    
    # Check if source directory exists
    if not os.path.exists(source_dir):
        print(f"Source directory {source_dir} does not exist.")
        print("Please create a dataset directory and place audio files there.")
        print("Alternatively, you can use the --create-dummy-data flag to create dummy data for testing.")
        return False
    
    # Get list of audio files
    audio_files = []
    
    # If subdirs provided, search in those
    if subdirs:
        for subdir in subdirs:
            subdir_path = os.path.join(source_dir, subdir)
            if os.path.exists(subdir_path) and os.path.isdir(subdir_path):
                for root, _, files in os.walk(subdir_path):
                    for file in files:
                        if file.endswith(('.wav', '.mp3')):
                            audio_files.append((os.path.join(root, file), file))
            else:
                print(f"Subdirectory {subdir_path} not found or not a directory.")
    else:
        # Search recursively in the source directory
        for root, _, files in os.walk(source_dir):
            for file in files:
                if file.endswith(('.wav', '.mp3')):
                    # Store both full path and filename
                    audio_files.append((os.path.join(root, file), file))
    
    if len(audio_files) == 0:
        print(f"No audio files found in {source_dir}")
        return False
    
    # Copy files
    copied_count = 0
    for source_path, filename in audio_files:
        dest_path = os.path.join(destination_dir, filename)
        
        # Skip if file already exists in destination
        if os.path.exists(dest_path):
            continue
        
        shutil.copy2(source_path, dest_path)
        copied_count += 1
    
    print(f"✅ Copied {copied_count} audio files to {destination_dir}")
    return True

def copy_annotation_files(source_dir, destination_dir):
    """
    Copy annotation files from source to destination directory
    
    Args:
        source_dir: Source directory in ADV_ML
        destination_dir: Destination directory in project
    """
    # Create destination directory if it doesn't exist
    os.makedirs(destination_dir, exist_ok=True)
    
    annotations_source = os.path.join(source_dir, "annotations")
    if not os.path.exists(annotations_source):
        print(f"Annotations directory {annotations_source} not found.")
        # Look in the parent directory
        annotations_source = os.path.join(os.path.dirname(source_dir), "annotations")
        if not os.path.exists(annotations_source):
            print("Could not find annotations directory.")
            return False
    
    print(f"Copying annotation files from {annotations_source} to {destination_dir}...")
    
    # Copy annotation files
    annotation_files = []
    for root, _, files in os.walk(annotations_source):
        for file in files:
            if file.endswith('.jams'):
                source_path = os.path.join(root, file)
                dest_path = os.path.join(destination_dir, file)
                
                # Skip if file already exists in destination
                if os.path.exists(dest_path):
                    continue
                
                shutil.copy2(source_path, dest_path)
                annotation_files.append(file)
    
    print(f"✅ Copied {len(annotation_files)} annotation files to {destination_dir}")
    return True

def create_dummy_data(output_dir, num_samples=10):
    """
    Create dummy audio data for testing when real data is not available
    
    Args:
        output_dir: Directory to save dummy data
        num_samples: Number of dummy samples to create
    """
    import numpy as np
    import soundfile as sf
    
    print(f"Creating {num_samples} dummy audio files in {output_dir}...")
    os.makedirs(output_dir, exist_ok=True)
    
    # Define chord names for dummy data
    chord_names = ['A', 'Am', 'C', 'D', 'Dm', 'E', 'Em', 'F', 'G']
    
    # Create dummy audio files
    for i in range(num_samples):
        # Select a random chord name
        chord = chord_names[i % len(chord_names)]
        
        # Create random audio data (1 second of noise at 22050Hz)
        duration = 1.0  # seconds
        sr = 22050  # sample rate
        audio_data = np.random.normal(0, 0.1, int(duration * sr))
        
        # Add a sine wave at a frequency corresponding to the chord
        # (just for some variation between samples)
        freq = 220 + (ord(chord[0]) - ord('A')) * 20  # map chord to frequency
        t = np.arange(0, duration, 1/sr)
        audio_data += 0.2 * np.sin(2 * np.pi * freq * t)
        
        # Save the file with the chord name as prefix
        file_path = os.path.join(output_dir, f"{chord}_dummy_{i+1}.wav")
        sf.write(file_path, audio_data, sr)
    
    print(f"✅ Created {num_samples} dummy audio files")
    return True

def main():
    """Main function to run the subproject"""
    parser = argparse.ArgumentParser(description="Guitar Chord Recognition with GuitarSet")
    parser.add_argument("--data-dir", type=str, default=DEFAULT_DATA_DIR,
                       help="Directory containing GuitarSet data")
    parser.add_argument("--annotations-dir", type=str, 
                       help="Directory containing annotation (.jams) files (optional, auto-detected if not specified)")
    parser.add_argument("--copy-data", action="store_true",
                       help="Copy raw data to project directory")
    parser.add_argument("--copy-annotations", action="store_true",
                       help="Copy annotation files to project directory")
    parser.add_argument("--create-dummy-data", action="store_true",
                       help="Create dummy data for testing when real data is not available")
    parser.add_argument("--num-dummy-samples", type=int, default=10,
                       help="Number of dummy samples to create")
    parser.add_argument("--preprocess-only", action="store_true",
                       help="Only preprocess data without training")
    parser.add_argument("--evaluate-only", action="store_true",
                       help="Only evaluate pretrained models without training")
    parser.add_argument("--batch-size", type=int, default=32,
                       help="Batch size for training")
    parser.add_argument("--epochs", type=int, default=100,
                       help="Number of epochs for training")
    parser.add_argument("--no-recursive", action="store_true",
                       help="Don't search recursively for audio files in subdirectories")
    parser.add_argument("--cowboy-chords", action="store_true", default=True,
                       help="Filter data to include only major cowboy chords (E, A, D, G, C, F) - enabled by default")
    parser.add_argument("--all-chords", action="store_true",
                       help="Use all detected chords instead of only cowboy chords")
    parser.add_argument("--disable-filter", action="store_true",
                       help="Disable chord filtering completely (debugging option)")
    parser.add_argument("--gpu", type=int, default=0,
                       help="GPU device ID to use for training (-1 for CPU)")
    parser.add_argument("--use-cpu", action="store_true",
                       help="Force using CPU even if GPU is available")
    
    args = parser.parse_args()
    
    # Create project directories
    setup_directories()
    
    # Determine data directory
    data_dir = args.data_dir
    raw_data_dir = os.path.join("./guitarset_chord_recognition/data", "raw")
    annotations_dir = os.path.join("./guitarset_chord_recognition/data", "annotations")
    
    # Copy or create data
    data_available = True
    if args.copy_data:
        # GuitarSet subdirectories to check for audio files
        guitarset_subdirs = ["audio", "audio_pickup", "audio_mic"]
        data_available = copy_raw_data(args.data_dir, raw_data_dir, subdirs=guitarset_subdirs)
        data_dir = raw_data_dir
        
        # Copy annotation files (required for segmentation)
        if args.annotations_dir:
            copy_annotation_files(args.annotations_dir, annotations_dir)
        else:
            copy_annotation_files(args.data_dir, annotations_dir)
    elif args.create_dummy_data:
        data_available = create_dummy_data(raw_data_dir, args.num_dummy_samples)
        data_dir = raw_data_dir
    elif not os.path.exists(data_dir):
        print(f"Error: Data directory {data_dir} does not exist.")
        print("Please use --copy-data to copy from another location or --create-dummy-data to create test data.")
        return
    
    # Check if data is available
    if not data_available:
        print("Data not available. Please check the data directory or use --create-dummy-data.")
        return
    
    # Preprocess data
    print("\n=== Data Preprocessing ===")
    try:
        # If all-chords is specified, disable cowboy-chords filtering
        use_cowboy_chords = args.cowboy_chords and not args.all_chords
        
        # If disable-filter is specified, override the chord filtering
        if args.disable_filter:
            use_cowboy_chords = None
            print("Chord filtering disabled for debugging")
        
        # Use explicitly provided annotations directory if available
        kwargs = {
            'disable_filter': args.disable_filter
        }
        if args.annotations_dir:
            print(f"Using specified annotations directory: {args.annotations_dir}")
            kwargs['annotations_dir'] = args.annotations_dir
        
        processed_files = process_dataset(data_dir, DEFAULT_PROCESSED_DIR, 
                                         recursive=not args.no_recursive,
                                         filter_chords=use_cowboy_chords,
                                         **kwargs)
        
        if len(processed_files) == 0:
            print("No files were processed. This could mean:")
            print("1. No audio files were found in the specified directory")
            print("2. There was an issue with the audio file format or processing")
            print("3. No files matched the chord filter criteria")
            print("4. No annotation files (.jams) were found")
            print("\nPlease make sure that annotation files are available in the appropriate location.")
            return
            
    except Exception as e:
        print(f"Error during preprocessing: {e}")
        print("If you don't have real data, try using --create-dummy-data to generate test data.")
        return
    
    # Stop here if only preprocessing
    if args.preprocess_only:
        print("\n✅ Data preprocessing completed!")
        return
    
    # Split into training and validation sets
    train_files, val_files = get_data_splits(processed_files, train_ratio=0.8)
    print(f"Training samples: {len(train_files)}")
    print(f"Validation samples: {len(val_files)}")
    
    # Determine device for training
    device = torch.device("cpu")
    if not args.use_cpu:
        if torch.cuda.is_available():
            if args.gpu >= 0 and args.gpu < torch.cuda.device_count():
                device = torch.device(f"cuda:{args.gpu}")
                print(f"Using GPU: {torch.cuda.get_device_name(device)}")
            else:
                device = torch.device("cuda:0")
                print(f"Using default GPU: {torch.cuda.get_device_name(device)}")
        else:
            print("No GPU available, using CPU instead")
    else:
        print("Forced CPU usage")

    # Create data loaders
    train_loader, val_loader, label_encoder = create_data_loaders(
        train_files, val_files, batch_size=args.batch_size
    )
    print(f"Number of classes: {len(label_encoder.classes_)}")
    print(f"Classes: {label_encoder.classes_}")
    
    # Train models or load pretrained models
    if not args.evaluate_only:
        model_results = train_models(train_loader, val_loader, label_encoder, 
                                     num_epochs=args.epochs, device=device)
    else:
        # Import necessary functions for loading models
        import pickle
        from guitarset_chord_recognition.models.cnn_model import CNNChordClassifier
        from guitarset_chord_recognition.models.bert_model import BERTInfusedChordClassifier
        from guitarset_chord_recognition.models.naive_bayes_model import NaiveBayesChordClassifier
        
        print("\n=== Loading pretrained models ===")
        
        num_classes = len(label_encoder.classes_)
        
        # Load CNN model
        cnn_model = CNNChordClassifier(num_classes=num_classes)
        cnn_save_path = os.path.join(DEFAULT_MODELS_DIR, "cnn_model.pt")
        if os.path.exists(cnn_save_path):
            cnn_model.load_state_dict(torch.load(cnn_save_path, map_location=device))
            cnn_model.to(device)
            print(f"Loaded CNN model from {cnn_save_path}")
        else:
            print(f"CNN model file {cnn_save_path} not found.")
            return
        
        # Load BERT model
        bert_model = BERTInfusedChordClassifier(num_classes=num_classes)
        bert_save_path = os.path.join(DEFAULT_MODELS_DIR, "bert_model.pt")
        if os.path.exists(bert_save_path):
            bert_model.load_state_dict(torch.load(bert_save_path, map_location=device))
            bert_model.to(device)
            print(f"Loaded BERT model from {bert_save_path}")
        else:
            print(f"BERT model file {bert_save_path} not found.")
            return
        
        # Load Naive Bayes model
        nb_save_path = os.path.join(DEFAULT_MODELS_DIR, "naive_bayes_model.pkl")
        if os.path.exists(nb_save_path):
            nb_model = NaiveBayesChordClassifier.load(nb_save_path)
            print(f"Loaded Naive Bayes model from {nb_save_path}")
        else:
            print(f"Naive Bayes model file {nb_save_path} not found.")
            return
        
        model_results = {
            'cnn': {'model': cnn_model, 'save_path': cnn_save_path},
            'bert': {'model': bert_model, 'save_path': bert_save_path},
            'naive_bayes': {'model': nb_model, 'save_path': nb_save_path}
        }
    
    # Evaluate models
    eval_results = evaluate_models(val_loader, model_results, label_encoder, device=device)
    
    # Compare models
    compare_models(eval_results, label_encoder)
    
    print("\n✅ Guitar chord recognition project completed!")

if __name__ == "__main__":
    main() 