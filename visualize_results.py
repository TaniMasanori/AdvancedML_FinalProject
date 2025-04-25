#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Acknowledgment: This project has been enhanced with the assistance of Claude AI by Anthropic
# Claude helped with code improvements, debugging, and implementation of the audio segmentation approach.

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import torch
from sklearn.metrics import confusion_matrix, classification_report

# Import local modules
from models.cnn_model import CNNModel
from models.bert_model import BERTModel
from utils.data_loader import load_test_data, prepare_data_loaders

def load_models():
    # Load CNN model
    cnn_model = CNNModel(num_classes=6)
    cnn_model.load_state_dict(torch.load('models/cnn_model.pt'))
    cnn_model.eval()
    
    # Load BERT model
    bert_model = BERTModel(num_classes=6)
    bert_model.load_state_dict(torch.load('models/bert_model.pt'))
    bert_model.eval()
    
    # Load Naive Bayes model
    with open('models/naive_bayes_model.pkl', 'rb') as f:
        nb_model = pickle.load(f)
    
    return cnn_model, bert_model, nb_model

def visualize_accuracy_comparison(accuracies):
    plt.figure(figsize=(10, 6))
    bars = plt.bar(['CNN', 'BERT', 'Naive Bayes'], accuracies, color=['#3498db', '#2ecc71', '#e74c3c'])
    plt.title('Model Accuracy Comparison', fontsize=16)
    plt.ylabel('Validation Accuracy', fontsize=14)
    plt.ylim(0, 1.0)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Add accuracy values on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                 f'{height:.2%}', ha='center', fontsize=12)
    
    plt.tight_layout()
    plt.savefig('results/accuracy_comparison.png', dpi=300)
    print("Saved accuracy comparison chart to results/accuracy_comparison.png")

def get_predictions(model, data_loader, model_type='cnn'):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Set model to evaluation mode
    if model_type != 'naive_bayes':
        model.to(device)
        model.eval()
    
    # Collect all predictions and true labels
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in data_loader:
            if model_type == 'naive_bayes':
                # For Naive Bayes, reshape and extract features from spectrogram data
                batch_size, channels, freq_bins, time_frames = inputs.shape
                features = []
                
                for i in range(batch_size):
                    # Extract frequency components
                    spec = inputs[i, 0].numpy()  # Take the first channel
                    
                    # Calculate spectral features (similar to the preprocessing step)
                    spec_mean = np.mean(spec)
                    spec_std = np.std(spec)
                    spec_max = np.max(spec)
                    
                    # Flatten and combine features
                    freq_features = np.mean(spec, axis=1)[:50]  # Take first 50 frequency bins
                    feature_vector = np.concatenate([freq_features, [spec_mean, spec_std, spec_max]])
                    features.append(feature_vector)
                
                predictions = model.predict(features)
                all_preds.extend(predictions)
                all_labels.extend(labels.numpy())
            else:
                # For CNN and BERT models
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                if model_type == 'cnn':
                    outputs = model(inputs)
                else:  # BERT model
                    outputs = model(inputs)
                
                _, predicted = torch.max(outputs, 1)
                
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
    
    return np.array(all_preds), np.array(all_labels)

def plot_confusion_matrix(y_true, y_pred, class_names, model_name):
    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(y_true, y_pred)
    
    # Normalize the confusion matrix
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    # Create a mask for zero values to avoid displaying them
    mask = cm == 0
    
    # Plot with seaborn
    sns.heatmap(cm_norm, annot=cm, fmt='d', cmap='Blues', xticklabels=class_names, 
                yticklabels=class_names, cbar=False, mask=mask)
    
    plt.title(f'Confusion Matrix - {model_name}', fontsize=16)
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()
    
    plt.savefig(f'results/confusion_matrix_{model_name.lower()}.png', dpi=300)
    print(f"Saved confusion matrix for {model_name} to results/confusion_matrix_{model_name.lower()}.png")

def main():
    # Create results directory if it doesn't exist
    os.makedirs('results', exist_ok=True)
    
    # Load models
    cnn_model, bert_model, nb_model = load_models()
    
    # Load test data
    test_loader = prepare_data_loaders(batch_size=32, test_only=True)
    
    # Get model accuracies
    with open('results/model_comparison.txt', 'r') as f:
        lines = f.readlines()
        accuracies = [float(line.split(': ')[1]) for line in lines if 'Accuracy' in line]
    
    # Visualize accuracy comparison
    visualize_accuracy_comparison(accuracies)
    
    # Get predictions for each model and plot confusion matrices
    class_names = ['A', 'C', 'D', 'E', 'F', 'G']
    
    # CNN model
    cnn_preds, cnn_true = get_predictions(cnn_model, test_loader, 'cnn')
    plot_confusion_matrix(cnn_true, cnn_preds, class_names, 'CNN')
    
    # BERT model
    bert_preds, bert_true = get_predictions(bert_model, test_loader, 'bert')
    plot_confusion_matrix(bert_true, bert_preds, class_names, 'BERT')
    
    # Naive Bayes model
    nb_preds, nb_true = get_predictions(nb_model, test_loader, 'naive_bayes')
    plot_confusion_matrix(nb_true, nb_preds, class_names, 'Naive_Bayes')
    
    print("Visualization complete!")

if __name__ == "__main__":
    main() 