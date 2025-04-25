#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simplified Chord prediction script for GuitarSet chord recognition
Loads a trained model and predicts chords from a hardcoded audio file.

Acknowledgment: This project has been enhanced with the assistance of Claude AI by Anthropic
Claude helped with code improvements, debugging, and implementation of the audio segmentation approach.
"""

import os
import sys
import torch
import pickle
import numpy as np
import matplotlib.pyplot as plt

# Add the project directory to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from guitarset_chord_recognition.utils.preprocessing import preprocess_audio_file
from guitarset_chord_recognition.models.cnn_model import CNNChordClassifier
from guitarset_chord_recognition.models.bert_model import BERTInfusedChordClassifier
from guitarset_chord_recognition.models.naive_bayes_model import NaiveBayesChordClassifier


# Hardcoded paths and settings

def load_models(models_dir=DEFAULT_MODELS_DIR):
    """
    Load trained models
    
    Args:
        models_dir: Directory containing trained models
        
    Returns:
        models: Dictionary of loaded models
        label_encoder: Fitted label encoder
    """
    # Check if models exist
    cnn_path = os.path.join(models_dir, "cnn_model.pt")
    bert_path = os.path.join(models_dir, "bert_model.pt")
    nb_path = os.path.join(models_dir, "naive_bayes_model.pkl")
    encoder_path = os.path.join(models_dir, "label_encoder.pkl")
    
    if not os.path.exists(encoder_path):
        raise FileNotFoundError(f"Label encoder not found at {encoder_path}. Please train models first.")
    
    # Load label encoder
    with open(encoder_path, 'rb') as f:
        label_encoder = pickle.load(f)
    
    # Determine device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # Dictionary to store models
    models = {}
    
    # Load CNN model
    if os.path.exists(cnn_path):
        print(f"Loading CNN model from {cnn_path}...")
        num_classes = len(label_encoder.classes_)
        cnn_model = CNNChordClassifier(num_classes=num_classes)
        cnn_model.load_state_dict(torch.load(cnn_path, map_location=device))
        cnn_model.to(device)
        cnn_model.eval()
        models['cnn'] = cnn_model
    
    # Load BERT model
    if os.path.exists(bert_path):
        print(f"Loading BERT model from {bert_path}...")
        num_classes = len(label_encoder.classes_)
        bert_model = BERTInfusedChordClassifier(num_classes=num_classes)
        bert_model.load_state_dict(torch.load(bert_path, map_location=device))
        bert_model.to(device)
        bert_model.eval()
        models['bert'] = bert_model
    
    # Load Naive Bayes model
    if os.path.exists(nb_path):
        print(f"Loading Naive Bayes model from {nb_path}...")
        nb_model = NaiveBayesChordClassifier.load(nb_path)
        models['naive_bayes'] = nb_model
    
    if not models:
        raise FileNotFoundError("No trained models found. Please train models first.")
    
    return models, label_encoder, device

def predict_chord(audio_path, models, label_encoder, device, cowboy_chords_only=False):
    """
    Predict chord from audio file using all available models
    
    Args:
        audio_path: Path to audio file
        models: Dictionary of loaded models
        label_encoder: Fitted label encoder
        device: Device to run inference on
        cowboy_chords_only: Whether to filter predictions to cowboy chords only
        
    Returns:
        predictions: Dictionary of predictions for each model
    """
    print(f"\nPredicting chord for {os.path.basename(audio_path)}...")
    
    # Define cowboy chords
    cowboy_chords = ['E', 'A', 'D', 'G', 'C', 'F']
    
    # Preprocess audio file
    spectrogram = preprocess_audio_file(audio_path)
    
    # Prepare for PyTorch models
    spectrogram_tensor = torch.FloatTensor(spectrogram).unsqueeze(0)  # Add batch dimension
    if len(spectrogram_tensor.shape) == 3:  # Add channel dimension if needed
        spectrogram_tensor = spectrogram_tensor.unsqueeze(0)
    spectrogram_tensor = spectrogram_tensor.to(device)
    
    # Store predictions
    predictions = {}
    
    # Predict with each model
    for model_name, model in models.items():
        if model_name in ['cnn', 'bert']:
            # PyTorch models
            with torch.no_grad():
                outputs = model(spectrogram_tensor)
                
                if cowboy_chords_only:
                    # Filter logits to only include cowboy chords
                    mask = torch.zeros_like(outputs, dtype=torch.bool)
                    for chord in cowboy_chords:
                        if chord in label_encoder.classes_:
                            idx = label_encoder.transform([chord])[0]
                            mask[:, idx] = True
                    
                    # Apply mask - set non-cowboy chord logits to very negative values
                    filtered_outputs = outputs.clone()
                    filtered_outputs[~mask] = -1e9
                    
                    # Compute probabilities from filtered logits
                    probs = torch.softmax(filtered_outputs, dim=1)
                else:
                    probs = torch.softmax(outputs, dim=1)
                
                confidence, predicted_idx = torch.max(probs, 1)
                
                # Get chord name
                chord_name = label_encoder.inverse_transform([predicted_idx.cpu().item()])[0]
                confidence = confidence.cpu().item()
        else:
            # Naive Bayes model
            if cowboy_chords_only:
                # For Naive Bayes, we need to get all predictions and filter
                all_probs = model.model.predict_proba(model.preprocess_features(spectrogram))[0]
                
                # Create filtered probabilities
                indices = [i for i, chord in enumerate(label_encoder.classes_) if chord in cowboy_chords]
                if not indices:  # If no cowboy chords in classes
                    chord_name = "unknown"
                    confidence = 0.0
                else:
                    # Get best cowboy chord
                    filtered_probs = np.zeros_like(all_probs)
                    filtered_probs[indices] = all_probs[indices]
                    
                    # Get best prediction from filtered probs
                    predicted_idx = np.argmax(filtered_probs)
                    confidence = filtered_probs[predicted_idx]
                    chord_name = label_encoder.inverse_transform([predicted_idx])[0]
            else:
                # Standard prediction
                chord_name, confidence = model.predict_chord(spectrogram)
        
        predictions[model_name] = {
            'chord': chord_name,
            'confidence': confidence
        }
    
    return predictions

def plot_predictions(predictions, output_path=None):
    """
    Visualize predictions
    
    Args:
        predictions: Dictionary of predictions
        output_path: Path to save visualization
    """
    # Extract data
    models = list(predictions.keys())
    chords = [pred['chord'] for pred in predictions.values()]
    confidences = [pred['confidence'] for pred in predictions.values()]
    
    # Plot
    plt.figure(figsize=(10, 6))
    bars = plt.bar(models, confidences, color=['#3498db', '#e74c3c', '#2ecc71'])
    
    # Add labels
    for i, bar in enumerate(bars):
        plt.text(
            bar.get_x() + bar.get_width()/2,
            bar.get_height() + 0.02,
            f"{chords[i]}\n{confidences[i]:.2f}",
            ha='center',
            va='bottom'
        )
    
    plt.ylim(0, 1.1)
    plt.title('Chord Predictions by Model')
    plt.ylabel('Confidence')
    plt.xlabel('Model')
    
    # Save or show
    if output_path:
        plt.savefig(output_path)
        print(f"Visualization saved to {output_path}")
    else:
        plt.show()

def main():
    parser = argparse.ArgumentParser(description="Predict guitar chords from audio files")
    parser.add_argument("--audio", type=str, required=True,
                       help="Path to audio file or directory")
    parser.add_argument("--models-dir", type=str, default=DEFAULT_MODELS_DIR,
                       help="Directory containing trained models")
    parser.add_argument("--results-dir", type=str, default=DEFAULT_RESULTS_DIR,
                       help="Directory to save results")
    parser.add_argument("--visualize", action="store_true",
                       help="Visualize predictions")
    parser.add_argument("--cowboy-chords-only", action="store_true",
                       help="Only show predictions for cowboy chords (E, A, D, G, C, F)")
    
    args = parser.parse_args()
    
    try:
        # Load models
        models, label_encoder, device = load_models(args.models_dir)
        
        # Filter for cowboy chords if requested
        if args.cowboy_chords_only:
            # Check if our label encoder contains any non-cowboy chords
            cowboy_chords = ['E', 'A', 'D', 'G', 'C', 'F']
            available_classes = list(label_encoder.classes_)
            
            if not all(chord in available_classes for chord in cowboy_chords):
                print("Warning: Some cowboy chords are not in the model's vocabulary.")
                print(f"Available classes: {available_classes}")
            
            print(f"Filtering predictions to cowboy chords only: {cowboy_chords}")
        
        # Check if audio path exists
        if not os.path.exists(args.audio):
            print(f"Error: {args.audio} does not exist.")
            return
        
        # Process single file or directory
        if os.path.isfile(args.audio):
            # Process single file
            predictions = predict_chord(args.audio, models, label_encoder, device, args.cowboy_chords_only)
            
            # Print predictions
            print("\nPredictions:")
            for model_name, pred in predictions.items():
                print(f"  {model_name.upper()}: {pred['chord']} (confidence: {pred['confidence']:.4f})")
            
            # Visualize if requested
            if args.visualize:
                os.makedirs(args.results_dir, exist_ok=True)
                output_path = os.path.join(args.results_dir, f"prediction_{os.path.basename(args.audio)}.png")
                plot_predictions(predictions, output_path)
        else:
            # Process directory
            audio_files = []
            for root, _, files in os.walk(args.audio):
                for file in files:
                    if file.endswith(('.wav', '.mp3')):
                        audio_files.append(os.path.join(root, file))
            
            if not audio_files:
                print(f"No audio files found in {args.audio}")
                return
            
            print(f"Found {len(audio_files)} audio files")
            
            # Process each file
            all_predictions = {}
            for audio_path in audio_files:
                predictions = predict_chord(audio_path, models, label_encoder, device, args.cowboy_chords_only)
                all_predictions[audio_path] = predictions
                
                # Print predictions
                print(f"\nPredictions for {os.path.basename(audio_path)}:")
                for model_name, pred in predictions.items():
                    print(f"  {model_name.upper()}: {pred['chord']} (confidence: {pred['confidence']:.4f})")
            
            # Visualize if requested
            if args.visualize:
                os.makedirs(args.results_dir, exist_ok=True)
                for audio_path, predictions in all_predictions.items():
                    output_path = os.path.join(args.results_dir, f"prediction_{os.path.basename(audio_path)}.png")
                    plot_predictions(predictions, output_path)
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main() 