#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Naive Bayes Model for Guitar Chord Recognition
Enhanced with improved frequency analysis based on musical acoustics
"""

import os
import numpy as np
import pickle
from scipy.fft import fft
from scipy.signal import find_peaks
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report
from tqdm import tqdm
# Acknowledgment: This project has been enhanced with the assistance of Claude AI by Anthropic
# Claude helped with code improvements, debugging, and implementation of the audio segmentation approach.


class NaiveBayesChordClassifier:
    """Naive Bayes classifier for guitar chord recognition with advanced frequency-based features"""
    
    def __init__(self):
        """Initialize the model"""
        self.model = GaussianNB()
        self.is_trained = False
        self.label_encoder = None
        
        # Frequencies of musical notes in base octave (C4, C#4, D4, etc.)
        self.base_frequencies = [261.63, 277.18, 293.66, 311.13, 329.63, 349.23, 
                               369.99, 392, 415.30, 440, 466.16, 493.88]
        
        # Common guitar open string frequencies (E2, A2, D3, G3, B3, E4)
        self.guitar_strings = [82.41, 110.00, 146.83, 196.00, 246.94, 329.63]
        
        # Create expanded frequency list including:
        # 1. Multiple octaves (lower and higher) important for guitar
        # 2. Add harmonics for each note (2x and 3x the fundamental frequency)
        self.frequencies = []
        
        # Add lower octave (important for guitar bass notes)
        self.frequencies.extend([f/2 for f in self.base_frequencies])
        
        # Add base octave
        self.frequencies.extend(self.base_frequencies)
        
        # Add higher octave (important for guitar high notes)
        self.frequencies.extend([f*2 for f in self.base_frequencies])
        
        # Add 2nd harmonic for base frequencies (2x fundamental)
        self.frequencies.extend([f*2 for f in self.base_frequencies])
        
        # Add 3rd harmonic for base frequencies (3x fundamental)
        self.frequencies.extend([f*3 for f in self.base_frequencies])
        
        # Add guitar open string frequencies if not already included
        for freq in self.guitar_strings:
            if freq not in self.frequencies:
                self.frequencies.append(freq)
        
        # Sort frequencies for better organization
        self.frequencies = sorted(list(set(self.frequencies)))
        
        # Create labels for the frequencies (for visualization)
        self.freq_labels = []
        octave_notes = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
        
        for freq in self.frequencies:
            # Find closest note
            for i, base_freq in enumerate(self.base_frequencies):
                # Base octave (C4-B4)
                if 0.97 * base_freq <= freq <= 1.03 * base_freq:
                    self.freq_labels.append(f"{octave_notes[i]}4")
                    break
                # Lower octave (C3-B3)
                elif 0.97 * base_freq/2 <= freq <= 1.03 * base_freq/2:
                    self.freq_labels.append(f"{octave_notes[i]}3")
                    break
                # Higher octave (C5-B5)
                elif 0.97 * base_freq*2 <= freq <= 1.03 * base_freq*2:
                    self.freq_labels.append(f"{octave_notes[i]}5")
                    break
                # 2nd harmonic
                elif 0.97 * base_freq*2 <= freq <= 1.03 * base_freq*2:
                    self.freq_labels.append(f"{octave_notes[i]}4_h2")
                    break
                # 3rd harmonic
                elif 0.97 * base_freq*3 <= freq <= 1.03 * base_freq*3:
                    self.freq_labels.append(f"{octave_notes[i]}4_h3")
                    break
            else:
                # If no match found, label as frequency value
                self.freq_labels.append(f"{freq:.1f}Hz")
        
        self.sr = 22050  # Sample rate
        
        # Define cowboy chords - standard guitar open chords
        self.cowboy_chords = ['E:maj', 'A:maj', 'D:maj', 'G:maj', 'C:maj', 'F:maj',
                             'E:min', 'A:min', 'D:min', 'G:min', 'C:min', 'F:min']
    
    def extract_advanced_features(self, signal, sr=22050):
        """
        Extract advanced features based on frequency analysis using FFT with
        enhanced musical acoustics knowledge
        
        Args:
            signal: Audio signal
            sr: Sample rate
            
        Returns:
            features: Array of features based on key frequencies and their harmonics
        """
        # Apply window function to reduce spectral leakage
        window = np.hamming(len(signal))
        windowed_signal = signal * window
        
        # Compute the FFT
        n = len(windowed_signal)
        fft_result = fft(windowed_signal)
        fft_magnitude = np.abs(fft_result[:n // 2])  # Magnitude of the FFT
        freqs = np.linspace(0, sr / 2, n // 2)  # Frequency array

        # Define the frequency range for search
        frequency_range = 5  # Frequency search range ±5Hz

        # Initialize feature array
        features = []

        # Calculate adaptive thresholds:
        # 1. Base threshold as in the utf_8gaussiannb implementation
        base_threshold = 15 * np.mean(fft_magnitude)
        
        # 2. Find spectral peaks for better detection of important frequencies
        peaks, _ = find_peaks(fft_magnitude, height=base_threshold*0.5, distance=sr/(2*frequency_range))
        peak_threshold = 0.5 * np.mean(fft_magnitude[peaks]) if len(peaks) > 0 else base_threshold
        
        # Use the more sensitive of the two thresholds
        threshold = min(base_threshold, peak_threshold)

        # Process each key frequency
        for freq in self.frequencies:
            # Find the index range around the key frequency
            low_idx = np.searchsorted(freqs, freq - frequency_range, side='left')
            high_idx = np.searchsorted(freqs, freq + frequency_range, side='right')
            
            # Find the maximum amplitude within the frequency range
            if low_idx < high_idx and high_idx < len(fft_magnitude):
                max_amplitude = np.max(fft_magnitude[low_idx:high_idx])
                # Compare with threshold and compute ratio if above threshold
                if max_amplitude > threshold:
                    # Add normalized feature value (exponential to emphasize stronger components)
                    feature_value = (max_amplitude / threshold) ** 1.5  # Emphasize stronger peaks
                    features.append(feature_value)
                else:
                    features.append(0.0)
            else:
                features.append(0.0)

        # Add additional features based on frequency relationships
        # These capture chord structures better than individual frequencies
        features = np.array(features)
        
        # Calculate normalized peak distribution (where along the spectrum energy is concentrated)
        if len(peaks) > 0:
            peak_positions = freqs[peaks]
            peak_values = fft_magnitude[peaks]
            weighted_pos = np.sum(peak_positions * peak_values) / np.sum(peak_values)
            # Normalize to 0-1 range based on Nyquist frequency
            norm_pos = weighted_pos / (sr/2)
            features = np.append(features, norm_pos)
        else:
            features = np.append(features, 0.0)
        
        # Add spectral centroid (brightness of sound)
        spec_sum = np.sum(fft_magnitude)
        if spec_sum > 0:
            centroid = np.sum(freqs * fft_magnitude) / spec_sum
            # Normalize to 0-1 range
            norm_centroid = centroid / (sr/2)
            features = np.append(features, norm_centroid)
        else:
            features = np.append(features, 0.0)
        
        # Add spectral flatness (ratio of geometric mean to arithmetic mean - tonal vs. noise)
        epsilon = 1e-10  # To avoid log(0) and division by zero
        geo_mean = np.exp(np.mean(np.log(fft_magnitude + epsilon)))
        arith_mean = np.mean(fft_magnitude + epsilon)
        if arith_mean > 0:
            flatness = geo_mean / arith_mean
            features = np.append(features, flatness)
        else:
            features = np.append(features, 0.0)

        return features
    
    def preprocess_features(self, spectrograms):
        """
        Preprocess spectrograms for Naive Bayes model
        
        Args:
            spectrograms: List of spectrograms or batch of spectrograms
            
        Returns:
            features: Frequency-based features
        """
        # Convert to numpy array if not already
        if isinstance(spectrograms, list):
            spectrograms = np.array(spectrograms)
        
        # If spectrograms are PyTorch tensors, convert to numpy
        if hasattr(spectrograms, 'numpy'):
            spectrograms = spectrograms.numpy()
        
        # Get original shape
        original_shape = spectrograms.shape
        
        # Handle different input shapes
        if len(original_shape) == 4:  # [batch_size, channels, height, width]
            batch_size = original_shape[0]
            # For batched data, extract features for each sample
            features = np.zeros((batch_size, len(self.frequencies) + 3))  # +3 for added features
            for i in range(batch_size):
                # Convert spectrogram back to time domain (approximately)
                # For simplicity, we'll use the first channel and take the mean across frequency
                signal = spectrograms[i, 0].mean(axis=0)  # Average across frequency axis
                features[i] = self.extract_advanced_features(signal, self.sr)
        elif len(original_shape) == 3:  # [channels, height, width]
            # Single sample, extract features
            signal = spectrograms[0].mean(axis=0)  # Average across frequency axis
            features = self.extract_advanced_features(signal, self.sr).reshape(1, -1)
        else:
            # Already in time domain
            features = self.extract_advanced_features(spectrograms, self.sr).reshape(1, -1)
        
        return features
    
    def train(self, train_loader, val_loader=None, save_path=None):
        """
        Train the Naive Bayes model
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader (optional)
            save_path: Path to save the model
            
        Returns:
            history: Training history
        """
        print("Extracting advanced frequency features for training...")
        X_train = []
        y_train = []
        
        # Extract features and labels from training data
        for inputs, labels in tqdm(train_loader):
            # Get numpy arrays from tensors
            inputs_np = inputs.numpy()
            labels_np = labels.numpy()
            
            # Preprocess features
            features = self.preprocess_features(inputs_np)
            
            # Append to training data
            X_train.append(features)
            y_train.append(labels_np)
        
        # Concatenate all batches
        X_train = np.vstack(X_train)
        y_train = np.concatenate(y_train)
        
        print(f"Training Naive Bayes model on {X_train.shape[0]} samples with {X_train.shape[1]} features")
        print(f"Feature vector includes {len(self.frequencies)} frequency components plus 3 spectral features")
        
        # Train the model
        self.model.fit(X_train, y_train)
        self.is_trained = True
        
        # Save label encoder from data loader for inference
        self.label_encoder = train_loader.dataset.label_encoder
        
        # Evaluate on training data
        train_preds = self.model.predict(X_train)
        train_acc = accuracy_score(y_train, train_preds)
        print(f"Training accuracy: {train_acc:.4f}")
        
        # Create history dictionary
        history = {
            'train_acc': train_acc,
            'val_acc': None
        }
        
        # Evaluate on validation data if provided
        if val_loader:
            print("Extracting validation features...")
            X_val = []
            y_val = []
            
            # Extract features and labels from validation data
            for inputs, labels in tqdm(val_loader):
                # Get numpy arrays from tensors
                inputs_np = inputs.numpy()
                labels_np = labels.numpy()
                
                # Preprocess features
                features = self.preprocess_features(inputs_np)
                
                # Append to validation data
                X_val.append(features)
                y_val.append(labels_np)
            
            # Concatenate all batches
            X_val = np.vstack(X_val)
            y_val = np.concatenate(y_val)
            
            # Evaluate on validation data
            val_preds = self.model.predict(X_val)
            val_acc = accuracy_score(y_val, val_preds)
            print(f"Validation accuracy: {val_acc:.4f}")
            
            # Add validation metrics to history
            history['val_acc'] = val_acc
            
            # Print classification report
            print("\nClassification Report (Validation):")
            report = classification_report(
                y_val, 
                val_preds, 
                target_names=self.label_encoder.classes_,
                zero_division=0
            )
            print(report)
            
            # Check accuracy on cowboy chords specifically
            if any(chord in self.cowboy_chords for chord in self.label_encoder.classes_):
                cowboy_indices = [i for i, label in enumerate(self.label_encoder.classes_) 
                                  if label in self.cowboy_chords]
                
                # Filter validation data to only include cowboy chords
                cowboy_mask = np.isin(y_val, cowboy_indices)
                if np.any(cowboy_mask):
                    cowboy_X = X_val[cowboy_mask]
                    cowboy_y = y_val[cowboy_mask]
                    
                    # Predict and calculate accuracy
                    cowboy_preds = self.model.predict(cowboy_X)
                    cowboy_acc = accuracy_score(cowboy_y, cowboy_preds)
                    print(f"\nCowboy chords accuracy: {cowboy_acc:.4f}")
                    history['cowboy_acc'] = cowboy_acc
        
        # Save model if path is provided
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            with open(save_path, 'wb') as f:
                pickle.dump({
                    'model': self.model,
                    'label_encoder': self.label_encoder,
                    'frequencies': self.frequencies,
                    'freq_labels': self.freq_labels,
                    'sr': self.sr
                }, f)
            print(f"Model saved to {save_path}")
        
        return history
    
    def predict(self, inputs):
        """
        Predict chord from input data
        
        Args:
            inputs: Input spectrogram(s)
            
        Returns:
            predictions: Predicted chord indices
            probabilities: Prediction probabilities
        """
        if not self.is_trained:
            raise ValueError("Model is not trained yet")
        
        # Preprocess features
        features = self.preprocess_features(inputs)
        
        # Predict
        predictions = self.model.predict(features)
        probabilities = self.model.predict_proba(features)
        
        return predictions, probabilities
    
    def predict_chord(self, spectrogram):
        """
        Predict chord from a single spectrogram
        
        Args:
            spectrogram: Input spectrogram
            
        Returns:
            chord_name: Predicted chord name
            confidence: Prediction confidence
        """
        if not self.is_trained:
            raise ValueError("Model is not trained yet")
        
        # Preprocess features
        features = self.preprocess_features(spectrogram)
        
        # Predict
        prediction = self.model.predict(features)[0]
        probabilities = self.model.predict_proba(features)[0]
        confidence = probabilities[prediction]
        
        # Get chord name
        chord_name = self.label_encoder.inverse_transform([prediction])[0]
        
        return chord_name, confidence
    
    def visualize_spectrum(self, signal, output_path=None):
        """
        Visualize frequency spectrum with key frequencies marked
        
        Args:
            signal: Audio signal
            output_path: Path to save visualization (optional)
        """
        import matplotlib.pyplot as plt
        
        # Apply window function for better visualization
        window = np.hamming(len(signal))
        windowed_signal = signal * window
        
        # Calculate the FFT
        n = len(windowed_signal)
        fft_result = fft(windowed_signal)
        fft_magnitude = np.abs(fft_result[:n//2])  # Magnitude of the FFT
        freqs = np.linspace(0, self.sr/2, n//2)  # Frequency array

        # Calculate threshold
        threshold = 15 * np.mean(fft_magnitude)
        
        # Find peaks
        peaks, _ = find_peaks(fft_magnitude, height=threshold*0.5, distance=self.sr/(2*10))

        # Plot the spectrum with log scale to show more detail
        plt.figure(figsize=(14, 6))
        plt.semilogy(freqs, fft_magnitude + 1e-10, label='Magnitude Spectrum')  # Add small offset to avoid log(0)
        plt.xlim(80, 1000)  # Focus on guitar frequency range
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Magnitude (log scale)')
        plt.title('Frequency Spectrum with Key Musical Frequencies')

        # Plot threshold line
        plt.axhline(y=threshold, color='green', linestyle='--', linewidth=1, 
                    label='Threshold (15x Mean Magnitude)')
        
        # Mark peaks
        if len(peaks) > 0:
            plt.plot(freqs[peaks], fft_magnitude[peaks], 'x', color='red', 
                     markersize=8, label='Detected Peaks')

        # Mark key frequencies (guitar strings and notes)
        # Guitar open strings (E2, A2, D3, G3, B3, E4)
        guitar_string_notes = ['E2', 'A2', 'D3', 'G3', 'B3', 'E4']
        for i, freq in enumerate(self.guitar_strings):
            plt.axvline(x=freq, color='purple', linestyle='-', linewidth=1, alpha=0.7)
            plt.text(freq, plt.ylim()[1]*0.8, guitar_string_notes[i], 
                     rotation=90, ha='right', color='purple', fontsize=8)

        # Mark base frequencies (C4-B4)
        note_names = ['C4', 'C#4', 'D4', 'D#4', 'E4', 'F4', 'F#4', 'G4', 'G#4', 'A4', 'A#4', 'B4']
        for i, freq in enumerate(self.base_frequencies):
            if 80 <= freq <= 1000:  # Only show in our visible range
                plt.axvline(x=freq, color='blue', linestyle='--', linewidth=0.8, alpha=0.5)
                plt.text(freq, plt.ylim()[1]*0.7, note_names[i], 
                         rotation=90, ha='right', color='blue', fontsize=8)

        plt.legend(loc='upper right')
        plt.grid(True, which="both", ls="--", alpha=0.7)
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path)
            plt.close()
        else:
            plt.show()
    
    def visualize_features(self, signal, output_path=None):
        """
        Visualize extracted features from audio signal
        
        Args:
            signal: Audio signal
            output_path: Path to save visualization (optional)
        """
        import matplotlib.pyplot as plt
        
        # Extract features
        features = self.extract_advanced_features(signal, self.sr)
        
        # Split frequency features and additional features
        freq_features = features[:len(self.frequencies)]
        extra_features = features[len(self.frequencies):]
        
        # Prepare simplified labels for frequency features (for readability)
        simplified_labels = []
        for label in self.freq_labels:
            # For harmonics, mark with '*'
            if '_h' in label:
                base_label = label.split('_')[0]
                harmonic = label.split('_')[1]
                simplified_labels.append(f"{base_label}*{harmonic[-1]}")
            else:
                simplified_labels.append(label)
        
        # Plot frequency features
        plt.figure(figsize=(16, 6))
        plt.subplot(1, 2, 1)
        bars = plt.bar(simplified_labels, freq_features)
        plt.title('Frequency Features')
        plt.ylabel('Feature Value')
        plt.xlabel('Musical Note')
        plt.xticks(rotation=90, fontsize=8)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        # Highlight non-zero features
        for i, bar in enumerate(bars):
            if bar.get_height() > 0.1:
                plt.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.05,
                        f'{bar.get_height():.2f}', ha='center', va='bottom', rotation=0, fontsize=8)
        
        # Plot additional features
        plt.subplot(1, 2, 2)
        extra_labels = ['Peak Position', 'Spectral Centroid', 'Spectral Flatness']
        plt.bar(extra_labels, extra_features)
        plt.title('Additional Spectral Features')
        plt.ylabel('Feature Value')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path)
            plt.close()
        else:
            plt.show()
    
    @classmethod
    def load(cls, model_path):
        """
        Load a trained model
        
        Args:
            model_path: Path to saved model
            
        Returns:
            model: Loaded model
        """
        with open(model_path, 'rb') as f:
            data = pickle.load(f)
        
        model = cls()
        model.model = data['model']
        model.label_encoder = data['label_encoder']
        # Load additional parameters if they exist
        if 'frequencies' in data:
            model.frequencies = data['frequencies']
        if 'freq_labels' in data:
            model.freq_labels = data['freq_labels']
        if 'sr' in data:
            model.sr = data['sr']
        model.is_trained = True
        
        return model 