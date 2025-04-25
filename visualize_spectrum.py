#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Utility script to visualize the frequency spectrum and features from audio files
Used for understanding the enhanced frequency-based Naive Bayes model

Acknowledgment: This project has been enhanced with the assistance of Claude AI by Anthropic
Claude helped with code improvements, debugging, and implementation of the audio segmentation approach.

"""

import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
import librosa
from scipy.fft import fft
from scipy.signal import find_peaks

# Add the project directory to sys.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from guitarset_chord_recognition.models.naive_bayes_model import NaiveBayesChordClassifier
from guitarset_chord_recognition.utils.preprocessing import preprocess_audio_file
from guitarset_chord_recognition.predict import load_models



def analyze_audio_spectrum(audio_path, output_dir=None):
    """
    Analyze the frequency spectrum of an audio file and visualize the features
    using the enhanced Naive Bayes model
    
    Args:
        audio_path: Path to the audio file
        output_dir: Directory to save visualizations (optional)
    """
    # Load audio
    y, sr = librosa.load(audio_path, sr=22050)
    
    # Create Naive Bayes model instance
    nb_model = NaiveBayesChordClassifier()
    
    # Create output directory if specified
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # Visualize raw waveform
    plt.figure(figsize=(12, 4))
    plt.plot(np.arange(len(y)) / sr, y)
    plt.title(f"Waveform: {os.path.basename(audio_path)}")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.grid(True, alpha=0.3)
    if output_dir:
        plt.savefig(os.path.join(output_dir, "waveform.png"))
        plt.close()
    else:
        plt.show()
    
    # Compute and visualize spectral features
    nb_model.visualize_spectrum(y, 
                              output_path=os.path.join(output_dir, "spectrum.png") if output_dir else None)
    
    # Visualize extracted features
    nb_model.visualize_features(y,
                              output_path=os.path.join(output_dir, "features.png") if output_dir else None)
    
    # Extract advanced features
    features = nb_model.extract_advanced_features(y)
    
    # Calculate additional audio metrics
    # Chromagram - shows the distribution of energy across the 12 pitch classes
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    
    # Visualize chromagram
    plt.figure(figsize=(12, 4))
    librosa.display.specshow(chroma, y_axis='chroma', x_axis='time')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Chromagram')
    if output_dir:
        plt.savefig(os.path.join(output_dir, "chromagram.png"))
        plt.close()
    else:
        plt.show()
    
    # Perform harmonic-percussive source separation for additional analysis
    y_harmonic, y_percussive = librosa.effects.hpss(y)
    
    # Plot the harmonic and percussive components
    plt.figure(figsize=(14, 8))
    plt.subplot(3, 1, 1)
    plt.plot(np.arange(len(y)) / sr, y)
    plt.title('Original Audio')
    plt.xlabel('Time (s)')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(3, 1, 2)
    plt.plot(np.arange(len(y_harmonic)) / sr, y_harmonic)
    plt.title('Harmonic Component (Tonal)')
    plt.xlabel('Time (s)')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(3, 1, 3)
    plt.plot(np.arange(len(y_percussive)) / sr, y_percussive)
    plt.title('Percussive Component (Transients)')
    plt.xlabel('Time (s)')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    if output_dir:
        plt.savefig(os.path.join(output_dir, "harmonic_percussive_separation.png"))
        plt.close()
    else:
        plt.show()
    
    # Calculate guitar-specific statistics
    print("\n=== Audio Analysis ===")
    print(f"Duration: {len(y)/sr:.2f} seconds")
    
    # Compare harmonic vs percussive energy (guitars should be more harmonic)
    harmonic_energy = np.sum(y_harmonic**2)
    percussive_energy = np.sum(y_percussive**2)
    total_energy = np.sum(y**2)
    
    print(f"Harmonic Energy: {100 * harmonic_energy / total_energy:.1f}%")
    print(f"Percussive Energy: {100 * percussive_energy / total_energy:.1f}%")
    
    # Identify the most prominent notes based on the chromagram
    chroma_mean = np.mean(chroma, axis=1)
    note_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    dominant_notes = [note_names[i] for i in np.argsort(chroma_mean)[::-1][:3]]
    
    print(f"Dominant Notes: {', '.join(dominant_notes)}")
    
    # Print most prominent frequency features
    freq_features = features[:len(nb_model.frequencies)]
    
    # Get the indices of the top 5 features by magnitude
    top_features_idx = np.argsort(freq_features)[::-1][:5]
    print("\nTop Frequency Components:")
    for idx in top_features_idx:
        if freq_features[idx] > 0.1:  # Only show significant components
            print(f"  {nb_model.freq_labels[idx]}: {freq_features[idx]:.2f}")
    
    # Print spectral features
    spectral_features = features[len(nb_model.frequencies):]
    spectral_names = ['Peak Distribution', 'Spectral Centroid', 'Spectral Flatness']
    print("\nSpectral Features:")
    for name, value in zip(spectral_names, spectral_features):
        print(f"  {name}: {value:.4f}")
    
    # Suggested chords based on prominent notes
    print("\nPossible Chords based on Dominant Notes:")
    chord_suggestions = []
    
    # Simple chord suggestion based on dominant notes
    major_chord_map = {
        'C': ['C', 'E', 'G'],
        'D': ['D', 'F#', 'A'],
        'E': ['E', 'G#', 'B'],
        'F': ['F', 'A', 'C'],
        'G': ['G', 'B', 'D'],
        'A': ['A', 'C#', 'E'],
        'B': ['B', 'D#', 'F#']
    }
    
    minor_chord_map = {
        'C': ['C', 'D#', 'G'],
        'D': ['D', 'F', 'A'],
        'E': ['E', 'G', 'B'],
        'F': ['F', 'G#', 'C'],
        'G': ['G', 'A#', 'D'],
        'A': ['A', 'C', 'E'],
        'B': ['B', 'D', 'F#']
    }
    
    # For each potential root note (top 3 dominant notes)
    for root in dominant_notes:
        root_name = root.replace('#', '')
        
        # Check if notes in major chord are present in dominant notes
        if root_name in major_chord_map:
            major_notes = set(major_chord_map[root_name])
            dominant_set = set(dominant_notes)
            
            # If at least 2 notes from the chord are in dominant notes
            if len(major_notes.intersection(dominant_set)) >= 2:
                chord_suggestions.append(f"{root} Major")
        
        # Check if notes in minor chord are present in dominant notes
        if root_name in minor_chord_map:
            minor_notes = set(minor_chord_map[root_name])
            dominant_set = set(dominant_notes)
            
            # If at least 2 notes from the chord are in dominant notes
            if len(minor_notes.intersection(dominant_set)) >= 2:
                chord_suggestions.append(f"{root} Minor")
    
    if chord_suggestions:
        print("  " + ", ".join(chord_suggestions))
    else:
        print("  No clear chord detected from dominant notes")
    
    return features

def predict_with_naive_bayes(audio_path, models_dir="./guitarset_chord_recognition/models"):
    """
    Predict chord using the enhanced Naive Bayes model
    
    Args:
        audio_path: Path to the audio file
        models_dir: Directory containing trained models
    """
    try:
        # Load models
        models, label_encoder, device = load_models(models_dir)
        
        # Check if Naive Bayes model is available
        if 'naive_bayes' not in models:
            print("Naive Bayes model not found in the models directory.")
            return
        
        # Preprocess audio file for prediction
        spectrogram = preprocess_audio_file(audio_path)
        
        # Get Naive Bayes model
        nb_model = models['naive_bayes']
        
        # Predict chord
        chord_name, confidence = nb_model.predict_chord(spectrogram)
        print(f"\n=== Naive Bayes Prediction ===")
        print(f"Audio: {os.path.basename(audio_path)}")
        print(f"Predicted Chord: {chord_name}")
        print(f"Confidence: {confidence:.4f}")
        
        # Get all probabilities
        _, probabilities = nb_model.predict(spectrogram)
        
        # Get top 5 predictions
        top_indices = np.argsort(probabilities[0])[::-1][:5]
        print("\nTop 5 predictions:")
        for i, idx in enumerate(top_indices):
            chord = label_encoder.inverse_transform([idx])[0]
            prob = probabilities[0][idx]
            print(f"{i+1}. {chord}: {prob:.4f}")
        
        # Create bar chart of top predictions
        plt.figure(figsize=(10, 6))
        chords = [label_encoder.inverse_transform([idx])[0] for idx in top_indices]
        probs = [probabilities[0][idx] for idx in top_indices]
        
        # Plot with color gradient based on probability
        colors = plt.cm.viridis(np.array(probs))
        bars = plt.bar(chords, probs, color=colors)
        
        plt.title('Top 5 Chord Predictions')
        plt.ylabel('Probability')
        plt.xlabel('Chord')
        plt.ylim(0, 1.0)
        plt.grid(axis='y', alpha=0.3)
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                    f'{height:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.show()
        
    except Exception as e:
        print(f"Error: {e}")

def main():
    parser = argparse.ArgumentParser(description="Visualize frequency spectrum and features from audio files")
    parser.add_argument("--audio", type=str, required=True,
                      help="Path to audio file")
    parser.add_argument("--output-dir", type=str, default=None,
                      help="Directory to save visualizations")
    parser.add_argument("--predict", action="store_true",
                      help="Predict chord using trained Naive Bayes model")
    parser.add_argument("--models-dir", type=str, default="./guitarset_chord_recognition/models",
                      help="Directory containing trained models (for prediction)")
    parser.add_argument("--compare-models", action="store_true",
                      help="Compare all available models (CNN, BERT, Naive Bayes)")
    
    args = parser.parse_args()
    
    # Check if audio path exists
    if not os.path.exists(args.audio):
        print(f"Error: Audio file {args.audio} does not exist.")
        return
    
    # Analyze spectrum and extract features
    print(f"Analyzing audio: {args.audio}")
    features = analyze_audio_spectrum(args.audio, args.output_dir)
    
    # Predict chord if requested
    if args.predict:
        predict_with_naive_bayes(args.audio, args.models_dir)
    
    # Compare all models if requested
    if args.compare_models:
        try:
            from guitarset_chord_recognition.predict import predict_chord
            
            # Load models
            models, label_encoder, device = load_models(args.models_dir)
            
            # Run prediction with all models
            predictions = predict_chord(args.audio, models, label_encoder, device, cowboy_chords_only=True)
            
            # Print predictions
            print("\n=== Model Comparison ===")
            print(f"Audio: {os.path.basename(args.audio)}")
            print("Predictions:")
            for model_name, pred in predictions.items():
                print(f"  {model_name.upper()}: {pred['chord']} (confidence: {pred['confidence']:.4f})")
            
            # Visualize predictions
            plt.figure(figsize=(10, 6))
            
            # Model names and colors
            model_names = list(predictions.keys())
            model_colors = {'cnn': '#3498db', 'bert': '#e74c3c', 'naive_bayes': '#2ecc71'}
            
            # Extract chords and confidences
            chords = [pred['chord'] for pred in predictions.values()]
            confidences = [pred['confidence'] for pred in predictions.values()]
            
            # Plot
            bars = plt.bar(
                model_names, 
                confidences, 
                color=[model_colors.get(name, '#9b59b6') for name in model_names]
            )
            
            # Add labels
            for i, bar in enumerate(bars):
                plt.text(
                    bar.get_x() + bar.get_width()/2,
                    bar.get_height() + 0.02,
                    f"{chords[i]}\n{confidences[i]:.3f}",
                    ha='center',
                    va='bottom'
                )
            
            plt.ylim(0, 1.1)
            plt.title('Chord Predictions by Model')
            plt.ylabel('Confidence')
            plt.xlabel('Model')
            plt.grid(axis='y', alpha=0.3)
            
            plt.tight_layout()
            plt.show()
            
        except Exception as e:
            print(f"Error comparing models: {e}")

if __name__ == "__main__":
    main() 