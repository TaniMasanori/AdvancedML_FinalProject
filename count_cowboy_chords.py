#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Count the number of cowboy chord samples in the GuitarSet dataset
"""

import os
import sys
import argparse
from collections import Counter
# Acknowledgment: This project has been enhanced with the assistance of Claude AI by Anthropic
# Claude helped with code improvements, debugging, and implementation of the audio segmentation approach.


# Add the project directory to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def count_chords(data_dir):
    """
    Count chord occurrences in the dataset
    
    Args:
        data_dir: Path to GuitarSet directory
    
    Returns:
        chord_counts: Counter object with chord counts
    """
    # Subdirectories to check
    subdirs = ["audio", "audio_pickup", "audio_mic"]
    
    # Cowboy chords to look for
    cowboy_chords = ['E', 'A', 'D', 'G', 'C', 'F']
    
    # Initialize counters
    all_chords = Counter()
    cowboy_chord_counts = Counter()
    
    for subdir in subdirs:
        subdir_path = os.path.join(data_dir, subdir)
        
        if not os.path.exists(subdir_path):
            print(f"Directory {subdir_path} not found.")
            continue
            
        print(f"\nScanning {subdir}:")
        for root, _, files in os.walk(subdir_path):
            for file in files:
                if file.endswith(('.wav', '.mp3')):
                    # Try to extract chord
                    parts = file.split('-')
                    if len(parts) >= 3:
                        chord = parts[2].split('_')[0]
                        chord = chord.strip()
                        all_chords[chord] += 1
                        
                        # Count cowboy chords
                        if chord in cowboy_chords:
                            cowboy_chord_counts[chord] += 1
    
    return all_chords, cowboy_chord_counts

def main():
    parser = argparse.ArgumentParser(description="Count cowboy chord samples in GuitarSet")
    parser.add_argument("--data-dir", type=str, default="/home/masa/adv_ml/guitarset",
                        help="Path to GuitarSet directory")
    
    args = parser.parse_args()
    
    # Check if directory exists
    if not os.path.exists(args.data_dir):
        print(f"Error: Directory {args.data_dir} does not exist.")
        return
    
    # Count chords
    all_chords, cowboy_chord_counts = count_chords(args.data_dir)
    
    # Print results
    print("\n=== Chord Distribution ===")
    print(f"Total unique chords found: {len(all_chords)}")
    
    print("\nCowboy Chord Distribution:")
    for chord in sorted(cowboy_chord_counts.keys()):
        print(f"- {chord}: {cowboy_chord_counts[chord]} files")
    
    print(f"\nTotal cowboy chord samples: {sum(cowboy_chord_counts.values())}")
    print(f"Total chord samples: {sum(all_chords.values())}")
    
    # Calculate percentage
    percentage = (sum(cowboy_chord_counts.values()) / sum(all_chords.values())) * 100
    print(f"Percentage of cowboy chords: {percentage:.2f}%")
    
    # Print all chords sorted by count
    print("\nAll chord types found (sorted by frequency):")
    for chord, count in all_chords.most_common():
        is_cowboy = "✓" if chord in cowboy_chord_counts else " "
        print(f"- {chord}: {count} files {is_cowboy}")

if __name__ == "__main__":
    main() 