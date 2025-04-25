#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Preprocessing utilities for GuitarSet chord recognition

Acknowledgment: This project has been enhanced with the assistance of Claude AI by Anthropic
Claude helped with code improvements, debugging, and implementation of the audio segmentation approach.
"""

import os
import numpy as np
import librosa
import pickle
import json
import glob
from tqdm import tqdm


def load_audio(file_path, sr=22050):
    """
    Load audio file and convert to mono if needed
    
    Args:
        file_path: Path to audio file
        sr: Target sample rate
        
    Returns:
        audio: Audio data as numpy array
        sr: Sample rate
    """
    audio, sr = librosa.load(file_path, sr=sr, mono=True)
    return audio, sr

def extract_mel_spectrogram(audio, sr=22050, n_fft=2048, hop_length=512, n_mels=128):
    """
    Extract MEL spectrogram from audio data
    
    Args:
        audio: Audio data as numpy array
        sr: Sample rate
        n_fft: FFT window size
        hop_length: Hop length for STFT
        n_mels: Number of MEL bands
        
    Returns:
        mel_spectrogram: MEL spectrogram
    """
    # Compute MEL spectrogram
    mel_spectrogram = librosa.feature.melspectrogram(
        y=audio, 
        sr=sr, 
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels
    )
    
    # Convert to logarithmic scale (dB)
    mel_spectrogram = librosa.power_to_db(mel_spectrogram, ref=np.max)
    
    return mel_spectrogram

def normalize_spectrogram(spectrogram):
    """
    Normalize spectrogram to range [0, 1]
    
    Args:
        spectrogram: Input spectrogram
        
    Returns:
        normalized_spectrogram: Normalized spectrogram
    """
    min_val = np.min(spectrogram)
    max_val = np.max(spectrogram)
    
    normalized_spectrogram = (spectrogram - min_val) / (max_val - min_val + 1e-8)
    return normalized_spectrogram

def resize_spectrogram(spectrogram, target_size=(128, 128)):
    """
    Resize spectrogram to target size
    
    Args:
        spectrogram: Input spectrogram
        target_size: Target size as (height, width)
        
    Returns:
        resized_spectrogram: Resized spectrogram
    """
    from skimage.transform import resize
    
    # Resize to target dimensions
    resized_spectrogram = resize(spectrogram, target_size, anti_aliasing=True)
    
    return resized_spectrogram

def preprocess_audio_file(file_path, target_size=(128, 128), sr=22050):
    """
    Preprocess audio file to extract normalized MEL spectrogram
    
    Args:
        file_path: Path to audio file
        target_size: Target spectrogram size
        sr: Target sample rate
        
    Returns:
        processed_spectrogram: Preprocessed spectrogram
    """
    # Load audio
    audio, sr = load_audio(file_path, sr=sr)
    
    # Extract MEL spectrogram
    mel_spec = extract_mel_spectrogram(audio, sr=sr)
    
    # Normalize
    norm_spec = normalize_spectrogram(mel_spec)
    
    # Resize if needed
    if target_size and norm_spec.shape != target_size:
        norm_spec = resize_spectrogram(norm_spec, target_size)
    
    return norm_spec

def preprocess_audio_segment(audio_segment, sr=22050, target_size=(128, 128)):
    """
    Preprocess an audio segment (numpy array) to extract normalized MEL spectrogram
    
    Args:
        audio_segment: Audio segment as numpy array
        sr: Sample rate
        target_size: Target spectrogram size
        
    Returns:
        processed_spectrogram: Preprocessed spectrogram
    """
    # Extract MEL spectrogram
    mel_spec = extract_mel_spectrogram(audio_segment, sr=sr)
    
    # Normalize
    norm_spec = normalize_spectrogram(mel_spec)
    
    # Resize if needed
    if target_size and norm_spec.shape != target_size:
        norm_spec = resize_spectrogram(norm_spec, target_size)
    
    return norm_spec

def extract_chords(jam_file):
    """
    Extract chord annotations from a .jams file
    
    Args:
        jam_file: Path to .jams file
    
    Returns:
        chord_list: List of (start_time, end_time, chord_label) tuples
    """
    with open(jam_file, "r") as f:
        jam_data = json.load(f)

    chord_list = []
    
    print(f"Examining chord annotations in {os.path.basename(jam_file)}")
    
    # There might be multiple chord annotations for different sources
    # We'll check all annotations with a chord namespace
    for annotation in jam_data["annotations"]:
        if annotation["namespace"] == "chord":
            annotation_label = annotation.get("annotation_metadata", {}).get("data_source", "unknown")
            print(f"  Found chord annotation from source: {annotation_label}")
            
            # For performed chords, there should be data points
            data_points = []
            
            for obs in annotation["data"]:
                start_time = obs["time"]
                duration = obs["duration"]
                end_time = start_time + duration
                chord_label = obs["value"]
                
                # Clean up the chord label - some might be in format like "N" for no chord
                if chord_label == "N" or chord_label.lower() == "no chord":
                    continue
                
                # Some chord labels might have complex structure like C:(3,5)
                # We'll simplify by taking just the root and quality
                if ":" in chord_label:
                    parts = chord_label.split(":")
                    root = parts[0]
                    
                    # Keep simplified chord name
                    if len(parts) > 1:
                        # If there's quality info
                        quality = parts[1]
                        # Handle common qualities
                        if "maj" in quality.lower() or "(3,5)" in quality:
                            chord_label = root  # Major chord
                        elif "min" in quality.lower() or "m" in quality or "(b3,5)" in quality:
                            chord_label = f"{root}m"  # Minor chord
                        elif "dim" in quality.lower() or "(b3,b5)" in quality:
                            chord_label = f"{root}dim"  # Diminished chord
                        elif "aug" in quality.lower() or "(3,#5)" in quality:
                            chord_label = f"{root}aug"  # Augmented chord
                        elif "7" in quality or "(3,5,b7)" in quality:
                            chord_label = f"{root}7"  # Dominant 7th
                        elif "9" in quality:
                            chord_label = f"{root}9"  # 9th chord
                        # If we still have a complex chord, just use the root
                        else:
                            chord_label = root
                    else:
                        chord_label = root
                
                # Extract the root note for matching with cowboy chords
                root_only = chord_label.split(':')[0].split('/')[0].split('(')[0]
                root_only = root_only[0] if len(root_only) > 0 else ''
                
                # Store both the full chord and the root-only version
                data_points.append((start_time, end_time, chord_label, root_only))
            
            if data_points:
                print(f"    Found {len(data_points)} chord segments")
                # Print first few chords for debugging
                for i, (start, end, chord, root) in enumerate(data_points[:5]):
                    print(f"    {i+1}. {chord} (root: {root}) from {start:.2f}s to {end:.2f}s")
                if len(data_points) > 5:
                    print(f"    ... (and {len(data_points)-5} more)")
            else:
                print("    No chord data points found in this annotation")
            
            # For chord_list, use only the start, end, and main chord label
            chord_list.extend([(start, end, chord) for start, end, chord, _ in data_points])

    if not chord_list:
        print("  No valid chord annotations found in this file")
    
    return chord_list

def sanitize_filename(name):
    """
    Create a safe filename from a chord name by removing invalid characters
    
    Args:
        name: Original chord name
        
    Returns:
        sanitized: Safe filename string
    """
    # Replace problem characters
    invalid_chars = [':', '*', '/', '\\', '?', '<', '>', '|', '"', "'"]
    sanitized = name
    for char in invalid_chars:
        sanitized = sanitized.replace(char, '_')
    return sanitized

def segment_audio(audio_file, chord_data, sr=22050, min_duration=0.5):
    """
    Splits an audio file into segments based on chord labels
    
    Args:
        audio_file: Path to audio file
        chord_data: List of (start_time, end_time, chord_label) tuples
        sr: Sample rate
        min_duration: Minimum segment duration in seconds to include
        
    Returns:
        segments: List of (audio_segment, chord) tuples
    """
    try:
        y, _ = librosa.load(audio_file, sr=sr)
    except Exception as e:
        print(f"Error loading audio file {audio_file}: {e}")
        return []
        
    segments = []

    for start, end, chord in chord_data:
        # Skip segments that are too short
        if end - start < min_duration:
            continue
            
        start_sample = int(start * sr)
        end_sample = int(end * sr)

        if end_sample <= start_sample or end_sample > len(y):
            continue  # Skip invalid segments

        segment = y[start_sample:end_sample]
        segments.append((segment, chord))

    return segments

def find_annotation_file(audio_file, annotations_dir):
    """
    Find the corresponding annotation file for an audio file
    
    Args:
        audio_file: Path to audio file
        annotations_dir: Directory containing annotation files
        
    Returns:
        annotation_file: Path to annotation file or None if not found
    """
    # Extract base name without extension
    basename = os.path.splitext(os.path.basename(audio_file))[0]
    
    # Search for matching annotation file
    annotation_files = glob.glob(os.path.join(annotations_dir, f"{basename}*.jams"))
    
    if annotation_files:
        return annotation_files[0]
    
    # If no direct match, try a more flexible search
    base_parts = basename.split('_')
    if len(base_parts) > 0:
        pattern = f"{base_parts[0]}*.jams"
        annotation_files = glob.glob(os.path.join(annotations_dir, pattern))
        if annotation_files:
            return annotation_files[0]
    
    return None

def get_annotation_dir(audio_dir):
    """
    Try to find the annotations directory based on the audio directory structure
    
    Args:
        audio_dir: Directory containing audio files
        
    Returns:
        annotations_dir: Path to annotations directory or None if not found
    """
    # Common patterns for GuitarSet
    potential_paths = [
        # Direct sibling to audio_dir (same level)
        os.path.join(os.path.dirname(audio_dir), "annotations"),
        
        # Check if the audio_dir itself is in a directory called "audio" and annotations is a sibling
        os.path.join(os.path.dirname(audio_dir) if os.path.basename(audio_dir) == "audio" 
                    else os.path.dirname(os.path.dirname(audio_dir)), "annotations"),
        
        # If audio_dir is inside guitarset_chord_recognition, try parent guitarset dir
        os.path.join(os.path.dirname(os.path.dirname(audio_dir)), "annotations"),
        
        # If audio_dir is a subdirectory of guitarset, look at the guitarset level
        os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(audio_dir))), "annotations"),
        
        # Exact path for the known guitarset location
        "/home/masa/adv_ml/guitarset/annotations",
        
        # These are the original paths
        os.path.join(audio_dir, "..", "annotations"),
        os.path.join(audio_dir, "..", "..", "annotations"),
    ]
    
    # For each path, print it and check if it exists
    for path in potential_paths:
        path = os.path.normpath(path)  # Normalize path to handle .. properly
        if os.path.exists(path) and os.path.isdir(path):
            print(f"Found annotations directory at: {path}")
            return path
    
    print("Checked the following paths for annotations directory:")
    for path in potential_paths:
        print(f"  - {os.path.normpath(path)}")
    
    return None

def process_dataset(input_dir, output_dir, label_map=None, recursive=True, filter_chords=None, annotations_dir=None, disable_filter=False):
    """
    Process all audio files in input directory and save preprocessed spectrograms
    Using segmentation based on JAMS annotation files
    
    Args:
        input_dir: Directory containing audio files
        output_dir: Directory to save preprocessed data
        label_map: Dictionary mapping filenames to labels (optional)
        recursive: Whether to search recursively for audio files in subdirectories
        filter_chords: List of chords to include (if None, include all chords)
        annotations_dir: Path to directory containing annotation files (if None, try to find automatically)
        disable_filter: If True, ignore filter_chords and include all chords (for debugging)
        
    Returns:
        processed_files: List of processed files
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Define cowboy chords with their various notation formats
    if filter_chords is True:
        filter_chords = ['E', 'A', 'D', 'G', 'C', 'F']
        # Also include explicit major notation forms that might appear in the dataset
        chord_patterns = [
            # Root notes
            'E', 'A', 'D', 'G', 'C', 'F',
            # Major chord notation variants
            'E:maj', 'A:maj', 'D:maj', 'G:maj', 'C:maj', 'F:maj',
            # Other common major notations
            'E(3,5)', 'A(3,5)', 'D(3,5)', 'G(3,5)', 'C(3,5)', 'F(3,5)'
        ]
        print(f"Filtering to major cowboy chords: E, A, D, G, C, F (including variations)")
    
    # Track all unique chord types found
    all_found_chords = set()
    root_notes_found = set()
    major_chords_found = set()
    
    # Get all audio files
    audio_files = []
    if recursive:
        # Walk through directories recursively
        for root, _, files in os.walk(input_dir):
            for file in files:
                if file.endswith(('.wav', '.mp3')):
                    audio_files.append(os.path.join(root, file))
    else:
        # Only get files in the top-level directory
        audio_files = [os.path.join(input_dir, f) for f in os.listdir(input_dir) 
                      if f.endswith(('.wav', '.mp3'))]
    
    processed_files = []
    filtered_count = 0
    skipped_count = 0
    
    print(f"Processing {len(audio_files)} audio files...")
    
    # Try to find annotations directory if not explicitly provided
    if not annotations_dir:
        annotations_dir = get_annotation_dir(input_dir)
    
    if annotations_dir:
        print(f"Using annotations directory: {annotations_dir}")
    else:
        print("ERROR: Annotations directory not found. This system requires annotation files (.jams).")
        print("Please make sure the annotations directory is available in a standard location.")
        print("Standard locations checked: 'annotations/' in parent directory or sibling to audio directory.")
        print("You can also specify the annotations directory with --annotations-dir")
        return []
    
    # If disabling filter, show what we would have filtered by
    if disable_filter and filter_chords:
        print(f"NOTE: Chord filtering is DISABLED (would have filtered to: {filter_chords})")
        filter_chords = None
    elif filter_chords:
        print(f"Filtering to include only these chords: {filter_chords}")
    
    # Get the maximum number of audio files to process for this run (for debugging)
    max_files = None  # Set to a smaller number while debugging, or None for all files
    
    for file_idx, file_path in enumerate(tqdm(audio_files)):
        if max_files and file_idx >= max_files:
            print(f"Stopping after {max_files} files (debugging mode)")
            break
            
        filename = os.path.basename(file_path)
        
        # Find corresponding annotation file
        annotation_file = find_annotation_file(file_path, annotations_dir)
        
        if not annotation_file:
            print(f"No annotation file found for {file_path}, skipping")
            skipped_count += 1
            continue
            
        # Extract chord annotations
        chord_annotations = extract_chords(annotation_file)
        
        if not chord_annotations:
            print(f"No chord annotations found in {annotation_file}, skipping")
            skipped_count += 1
            continue
        
        # Segment audio based on annotations
        audio_segments = segment_audio(file_path, chord_annotations)
        
        if not audio_segments:
            print(f"No valid segments found in {file_path}, skipping")
            skipped_count += 1
            continue
        
        # Process each segment
        segment_count = 0
        for i, (segment, chord) in enumerate(audio_segments):
            # Keep track of all chord types found
            all_found_chords.add(chord)
            
            # Extract root note (just the letter part) for cowboy chord matching
            root_note = chord[0] if chord else ''
            root_notes_found.add(root_note)
            
            # Check if it's a major chord notation
            is_major = (
                # Root-only implies major
                (len(chord) == 1) or 
                # Explicit major notations
                ":maj" in chord or 
                # Triad notation (3rd and 5th)
                "(3,5)" in chord or 
                # No minor/diminished/etc. indicator
                (not any(x in chord for x in ['m', 'dim', 'aug', '7', '9']))
            )
            if is_major and root_note in ['E', 'A', 'D', 'G', 'C', 'F']:
                major_chords_found.add(f"{root_note}:maj")
            
            # Skip if we're filtering chords and this chord is not in our list
            chord_matches_filter = False
            if filter_chords:
                # Check if the chord is a major cowboy chord
                if root_note in filter_chords and is_major:
                    chord_matches_filter = True
                # Also check exact matches for explicit formats
                elif chord in chord_patterns:
                    chord_matches_filter = True
                
                if not chord_matches_filter:
                    filtered_count += 1
                    continue
            
            try:
                # Process audio segment
                spectrogram = preprocess_audio_segment(segment)
                
                # Use consistent chord labeling: simplify all formats to root note only
                # for the cowboy chords (since they're all major anyway)
                simplified_chord = root_note if root_note in ['E', 'A', 'D', 'G', 'C', 'F'] and is_major else chord
                
                # Create safe filename
                safe_chord = sanitize_filename(simplified_chord)
                
                # Save as pickle file
                output_filename = f"{safe_chord}_{os.path.splitext(filename)[0]}_seg{i}.pkl"
                output_path = os.path.join(output_dir, output_filename)
                
                with open(output_path, 'wb') as f:
                    pickle.dump(spectrogram, f)
                
                processed_files.append((output_path, simplified_chord))
                segment_count += 1
            except Exception as e:
                print(f"Error processing segment {i} from {file_path}: {e}")
                continue
        
        if segment_count > 0:
            print(f"Processed {segment_count} segments from {filename}")
    
    if filtered_count > 0:
        print(f"Filtering applied: {filtered_count} segments excluded")
    
    if skipped_count > 0:
        print(f"Skipped {skipped_count} files due to missing annotations or processing errors")
    
    # Print summary of all chord types found
    print("\nAll unique chord types found in the dataset:")
    sorted_chords = sorted(list(all_found_chords))
    print(", ".join(sorted_chords))
    
    # Also show root notes found
    print("\nRoot notes found in the dataset:")
    sorted_roots = sorted(list(root_notes_found))
    print(", ".join(sorted_roots))
    
    # Show major cowboy chords found
    print("\nMajor cowboy chords found in the dataset:")
    sorted_majors = sorted(list(major_chords_found))
    print(", ".join(sorted_majors))
    
    if filter_chords:
        # Count how many segments we kept for each chord
        chord_counts = {}
        for _, chord in processed_files:
            chord_counts[chord] = chord_counts.get(chord, 0) + 1
        
        print("\nChord distribution in processed segments:")
        for chord, count in sorted(chord_counts.items()):
            print(f"  {chord}: {count} segments")
    
    print(f"Processing complete. {len(processed_files)} segments saved to {output_dir}")
    return processed_files

def get_data_splits(processed_files, train_ratio=0.8, shuffle=True):
    """
    Split processed files into training and validation sets
    
    Args:
        processed_files: List of (file_path, label) tuples
        train_ratio: Ratio of training data
        shuffle: Whether to shuffle data before splitting
        
    Returns:
        train_files: Training files
        val_files: Validation files
    """
    import random
    
    if shuffle:
        random.shuffle(processed_files)
    
    split_idx = int(len(processed_files) * train_ratio)
    train_files = processed_files[:split_idx]
    val_files = processed_files[split_idx:]
    
    return train_files, val_files 