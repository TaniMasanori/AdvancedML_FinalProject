#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BERT-infused Model for Guitar Chord Recognition
This model uses a BERT-like architecture (transformer encoder) for processing spectrogram data
Based on an improved implementation from guitar_train_bert.py
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
# Acknowledgment: This project has been enhanced with the assistance of Claude AI by Anthropic
# Claude helped with code improvements, debugging, and implementation of the audio segmentation approach.


class PositionalEncoding(nn.Module):
    """Positional encoding for transformer architecture"""
    
    def __init__(self, d_model, max_len=1024):
        super(PositionalEncoding, self).__init__()
        
        # Create positional encoding
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # Register buffer to save in state_dict but not as model parameters
        self.register_buffer('pe', pe.unsqueeze(0))
        
    def forward(self, x):
        # x: [batch_size, seq_len, d_model]
        # Ensure we don't try to access beyond pe's first dimension
        seq_len = x.size(1)
        return x + self.pe[:, :seq_len]

class BERTInfusedChordClassifier(nn.Module):
    """BERT-inspired model with multi-head attention for chord recognition"""
    
    def __init__(self, num_classes=6, num_heads=8):
        super(BERTInfusedChordClassifier, self).__init__()
        
        # Feature extraction (CNN layers)
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        
        # Define embedding dimension based on feature extractor output
        self.embed_dim = 128
        
        # Multi-head attention (using PyTorch's built-in implementation)
        self.multihead_attn = nn.MultiheadAttention(
            embed_dim=self.embed_dim,
            num_heads=num_heads,
            batch_first=True
        )
        
        # Positional encoding for sequence of tokens
        self.pos_encoder = PositionalEncoding(d_model=self.embed_dim, max_len=1024)
        
        # PyTorch's LayerNorm for feature dimensions
        self.layer_norm1 = nn.LayerNorm(self.embed_dim)
        self.layer_norm2 = nn.LayerNorm(self.embed_dim)
        
        # Feed-forward network (similar to BERT)
        self.feed_forward = nn.Sequential(
            nn.Linear(self.embed_dim, self.embed_dim * 4),
            nn.ReLU(),
            nn.Linear(self.embed_dim * 4, self.embed_dim)
        )
        
        # Global pooling and classification
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(0.5)
        self.classifier = nn.Linear(self.embed_dim, num_classes)
    
    def forward(self, x):
        # Feature extraction
        x = self.feature_extractor(x)  # [batch_size, channels, height, width]
        
        # Reshape for multi-head attention: [batch_size, seq_len, embed_dim]
        # Flatten spatial dimensions into sequence length
        batch_size, channels, height, width = x.size()
        x_flat = x.view(batch_size, channels, height * width)  # [B, C, H*W]
        x_flat = x_flat.permute(0, 2, 1)  # [B, H*W, C]
        
        # Apply positional encoding
        x_pos = self.pos_encoder(x_flat)
        
        # Apply multi-head attention
        attn_output, _ = self.multihead_attn(
            query=x_pos,
            key=x_pos,
            value=x_pos
        )
        
        # First residual connection and layer norm
        x_norm = self.layer_norm1(x_pos + attn_output)
        
        # Feed-forward network
        ff_output = self.feed_forward(x_norm)
        
        # Second residual connection and layer norm
        x_out = self.layer_norm2(x_norm + ff_output)
        
        # Reshape back to original format for compatibility with the rest of the network
        x_reshaped = x_out.permute(0, 2, 1)  # [B, C, H*W]
        x_reshaped = x_reshaped.view(batch_size, channels, height, width)  # [B, C, H, W]
        
        # Global pooling and classification
        x_pooled = self.global_pool(x_reshaped)
        x_pooled = x_pooled.view(x_pooled.size(0), -1)  # Flatten
        x_pooled = self.dropout(x_pooled)
        x_out = self.classifier(x_pooled)
        
        return x_out

def train_model(model, train_loader, val_loader, 
                num_epochs=100, learning_rate=0.001,
                device=None, save_path=None):
    """
    Train the BERT-infused model
    
    Args:
        model: Model to train
        train_loader: Training data loader
        val_loader: Validation data loader
        num_epochs: Number of epochs to train for
        learning_rate: Learning rate
        device: Device to train on (CPU/GPU)
        save_path: Path to save the model
        
    Returns:
        model: Trained model
        history: Training history
    """
    # Set device
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    model = model.to(device)
    
    # Print device information
    if device.type == 'cuda':
        print(f"Training on GPU: {torch.cuda.get_device_name(device)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(device).total_memory / 1e9:.2f} GB")
        # Enable cuDNN benchmark for best performance
        torch.backends.cudnn.benchmark = True
    else:
        print("Training on CPU")
    
    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    
    # Mixed precision training setup
    scaler = None
    if device.type == 'cuda' and torch.cuda.is_available():
        try:
            from torch.cuda.amp import GradScaler, autocast
            scaler = GradScaler()
            print("Using mixed precision training for faster performance")
        except ImportError:
            print("Mixed precision training not available, using full precision")
    
    # Use AdamW with weight decay for better regularization
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    
    # Training history
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }
    
    # Best model tracking
    best_val_loss = float('inf')
    patience = 10  # Early stopping patience
    patience_counter = 0
    
    # Training loop
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for inputs, labels in train_loader:
            # Move tensors to device with non_blocking for faster GPU transfers
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            
            # Zero the gradients
            optimizer.zero_grad(set_to_none=True)  # More efficient than zero_grad()
            
            # Mixed precision forward pass if available
            if scaler is not None:
                with autocast():
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                
                # Backward pass with gradient scaling
                scaler.scale(loss).backward()
                
                # Gradient clipping with scaling
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                
                # Optimizer step with scaling
                scaler.step(optimizer)
                scaler.update()
            else:
                # Regular forward pass
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                # Backward pass and optimize
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                
                optimizer.step()
            
            # Track loss and accuracy
            train_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            train_correct += (predicted == labels).sum().item()
            train_total += labels.size(0)
        
        # Calculate average training loss and accuracy
        train_loss = train_loss / len(train_loader.dataset)
        train_acc = train_correct / train_total
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                # Move tensors to device with non_blocking for faster GPU transfers
                inputs = inputs.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                
                # Forward pass
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                # Track loss and accuracy
                val_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs, 1)
                val_correct += (predicted == labels).sum().item()
                val_total += labels.size(0)
        
        # Calculate average validation loss and accuracy
        val_loss = val_loss / len(val_loader.dataset)
        val_acc = val_correct / val_total
        
        # Update scheduler
        scheduler.step()
        
        # Save best model
        if val_loss < best_val_loss and save_path:
            best_val_loss = val_loss
            torch.save(model.state_dict(), save_path)
            print(f"Saved best model with val_loss: {val_loss:.4f}")
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping triggered after {epoch+1} epochs")
                break
        
        # Update history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        # Print progress
        print(f"Epoch {epoch+1}/{num_epochs} | "
              f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")
    
    # Load best model if saved
    if save_path and os.path.exists(save_path):
        model.load_state_dict(torch.load(save_path))
        print(f"Loaded best model from {save_path}")
    
    return model, history 