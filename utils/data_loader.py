#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Data loader utilities for GuitarSet chord recognition

Acknowledgment: This project has been enhanced with the assistance of Claude AI by Anthropic
Claude helped with code improvements, debugging, and implementation of the audio segmentation approach.
"""

import os
import pickle
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder

class GuitarChordDataset(Dataset):
    """Dataset class for Guitar chord recognition"""
    
    def __init__(self, file_paths, transform=None):
        """
        Initialize the dataset
        
        Args:
            file_paths: List of (file_path, label) tuples
            transform: Optional transform to apply to the data
        """
        self.file_paths = file_paths
        self.transform = transform
        
        # Extract labels
        self.labels = [label for _, label in file_paths]
        
        # Encode labels
        self.label_encoder = LabelEncoder()
        self.encoded_labels = self.label_encoder.fit_transform(self.labels)
    
    def __len__(self):
        return len(self.file_paths)
    
    def __getitem__(self, idx):
        file_path, _ = self.file_paths[idx]
        
        # Load spectrogram
        with open(file_path, 'rb') as f:
            spectrogram = pickle.load(f)
        
        # Convert to tensor
        spectrogram = torch.FloatTensor(spectrogram)
        
        # Add channel dimension if needed
        if len(spectrogram.shape) == 2:
            spectrogram = spectrogram.unsqueeze(0)  # Add channel dimension
        
        # Apply transform if specified
        if self.transform:
            spectrogram = self.transform(spectrogram)
        
        # Get label
        label = torch.tensor(self.encoded_labels[idx], dtype=torch.long)
        
        return spectrogram, label
    
    def get_label_encoder(self):
        """Get the label encoder"""
        return self.label_encoder

def create_data_loaders(train_files, val_files, batch_size=32, num_workers=4):
    """
    Create PyTorch DataLoaders for training and validation
    
    Args:
        train_files: List of (file_path, label) tuples for training
        val_files: List of (file_path, label) tuples for validation
        batch_size: Batch size
        num_workers: Number of workers for data loading
        
    Returns:
        train_loader: Training data loader
        val_loader: Validation data loader
        label_encoder: Fitted label encoder
    """
    # Create datasets
    train_dataset = GuitarChordDataset(train_files)
    val_dataset = GuitarChordDataset(val_files)
    
    # Get label encoder from training dataset
    label_encoder = train_dataset.get_label_encoder()
    
    # Update validation dataset to use same label encoder
    val_dataset.label_encoder = label_encoder
    val_dataset.encoded_labels = label_encoder.transform(val_dataset.labels)
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )
    
    return train_loader, val_loader, label_encoder 