#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Model Comparison for Guitar Chord Recognition
This script compares the performance of CNN, BERT-infused, and Naive Bayes models
"""

import os
import numpy as np
import torch
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from tqdm import tqdm
# Acknowledgment: This project has been enhanced with the assistance of Claude AI by Anthropic
# Claude helped with code improvements, debugging, and implementation of the audio segmentation approach.


# Default paths
DEFAULT_PROCESSED_DIR = "./guitarset_chord_recognition/data/processed"
DEFAULT_MODELS_DIR = "./guitarset_chord_recognition/models"
DEFAULT_RESULTS_DIR = "./guitarset_chord_recognition/results"

def setup_directories():
    """Create necessary directories"""
    os.makedirs(DEFAULT_PROCESSED_DIR, exist_ok=True)
    os.makedirs(DEFAULT_MODELS_DIR, exist_ok=True)
    os.makedirs(DEFAULT_RESULTS_DIR, exist_ok=True)
    os.makedirs(os.path.join("./guitarset_chord_recognition/data", "raw"), exist_ok=True)

def train_models(train_loader, val_loader, label_encoder, num_epochs=100, device=None):
    """Train all three models"""
    from guitarset_chord_recognition.models.cnn_model import CNNChordClassifier, train_model as train_cnn
    from guitarset_chord_recognition.models.bert_model import BERTInfusedChordClassifier, train_model as train_bert
    from guitarset_chord_recognition.models.naive_bayes_model import NaiveBayesChordClassifier
    
    # Set default device if not provided
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    num_classes = len(label_encoder.classes_)
    results = {}
    
    # Train CNN model
    print("\n=== Training CNN Model ===")
    cnn_model = CNNChordClassifier(num_classes=num_classes)
    cnn_save_path = os.path.join(DEFAULT_MODELS_DIR, "cnn_model.pt")
    cnn_model, cnn_history = train_cnn(
        cnn_model, train_loader, val_loader, 
        num_epochs=num_epochs, device=device, save_path=cnn_save_path
    )
    results['cnn'] = {
        'model': cnn_model,
        'history': cnn_history,
        'save_path': cnn_save_path
    }
    
    # Train BERT-infused model
    print("\n=== Training BERT-infused Model ===")
    bert_model = BERTInfusedChordClassifier(num_classes=num_classes)
    bert_save_path = os.path.join(DEFAULT_MODELS_DIR, "bert_model.pt")
    bert_model, bert_history = train_bert(
        bert_model, train_loader, val_loader, 
        num_epochs=num_epochs, device=device, save_path=bert_save_path
    )
    results['bert'] = {
        'model': bert_model,
        'history': bert_history,
        'save_path': bert_save_path
    }
    
    # Train Naive Bayes model
    print("\n=== Training Naive Bayes Model ===")
    nb_model = NaiveBayesChordClassifier()
    nb_save_path = os.path.join(DEFAULT_MODELS_DIR, "naive_bayes_model.pkl")
    nb_history = nb_model.train(train_loader, val_loader, save_path=nb_save_path)
    results['naive_bayes'] = {
        'model': nb_model,
        'history': nb_history,
        'save_path': nb_save_path
    }
    
    # Save label encoder
    encoder_path = os.path.join(DEFAULT_MODELS_DIR, "label_encoder.pkl")
    with open(encoder_path, 'wb') as f:
        pickle.dump(label_encoder, f)
    
    return results

def evaluate_models(val_loader, model_results, label_encoder, device=None):
    """Evaluate and compare all models"""
    # Set default device if not provided
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    eval_results = {}
    
    # Extract class names
    class_names = label_encoder.classes_
    
    # Evaluate each model
    for model_name, model_data in model_results.items():
        print(f"\n=== Evaluating {model_name} Model ===")
        
        model = model_data['model']
        
        # Initialize lists for predictions and true labels
        all_preds = []
        all_true = []
        
        # Evaluate
        if model_name in ['cnn', 'bert']:
            # PyTorch models
            model.eval()
            with torch.no_grad():
                for inputs, labels in tqdm(val_loader):
                    inputs = inputs.to(device)
                    labels = labels.to(device)
                    
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    
                    all_preds.extend(preds.cpu().numpy())
                    all_true.extend(labels.cpu().numpy())
        else:
            # Naive Bayes model
            for inputs, labels in tqdm(val_loader):
                features = model.preprocess_features(inputs)
                preds = model.model.predict(features)
                
                all_preds.extend(preds)
                all_true.extend(labels.numpy())
        
        # Calculate accuracy
        accuracy = accuracy_score(all_true, all_preds)
        print(f"Accuracy: {accuracy:.4f}")
        
        # Generate classification report
        report = classification_report(
            all_true, all_preds, 
            target_names=class_names, 
            output_dict=True
        )
        
        # Generate confusion matrix
        cm = confusion_matrix(all_true, all_preds)
        
        # Store results
        eval_results[model_name] = {
            'accuracy': accuracy,
            'report': report,
            'confusion_matrix': cm,
            'predictions': all_preds,
            'true_labels': all_true
        }
        
        # Plot confusion matrix
        plt.figure(figsize=(12, 10))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=class_names, yticklabels=class_names)
        plt.title(f'{model_name.upper()} Model Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.tight_layout()
        plt.savefig(os.path.join(DEFAULT_RESULTS_DIR, f"{model_name}_confusion_matrix.png"))
    
    return eval_results

def compare_models(eval_results, label_encoder):
    """Compare model performance and generate visualizations"""
    import pandas as pd
    
    print("\n=== Model Comparison ===")
    
    # Extract accuracies
    accuracies = {model_name: results['accuracy'] 
                 for model_name, results in eval_results.items()}
    
    # Print accuracies
    for model_name, acc in accuracies.items():
        print(f"{model_name.upper()} Model Accuracy: {acc:.4f}")
    
    # Plot accuracy comparison
    plt.figure(figsize=(10, 6))
    plt.bar(accuracies.keys(), accuracies.values())
    plt.title('Model Accuracy Comparison')
    plt.xlabel('Model')
    plt.ylabel('Accuracy')
    plt.ylim(0, 1)
    for i, (model, acc) in enumerate(accuracies.items()):
        plt.text(i, acc + 0.01, f'{acc:.4f}', ha='center')
    plt.tight_layout()
    plt.savefig(os.path.join(DEFAULT_RESULTS_DIR, "accuracy_comparison.png"))
    
    # Compare class-wise performance
    class_names = label_encoder.classes_
    class_f1_scores = {model_name: {class_name: results['report'][class_name]['f1-score'] 
                                   for class_name in class_names}
                      for model_name, results in eval_results.items()}
    
    # Create comparison DataFrame
    df_f1 = pd.DataFrame(class_f1_scores)
    
    # Plot class-wise F1 scores
    plt.figure(figsize=(14, 10))
    df_f1.plot(kind='bar', figsize=(14, 8))
    plt.title('F1 Score Comparison by Chord Class')
    plt.xlabel('Chord Class')
    plt.ylabel('F1 Score')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(DEFAULT_RESULTS_DIR, "f1_score_comparison.png"))
    
    # Save detailed comparison results
    with open(os.path.join(DEFAULT_RESULTS_DIR, "model_comparison.pkl"), 'wb') as f:
        pickle.dump({
            'accuracies': accuracies,
            'class_f1_scores': class_f1_scores,
            'full_results': eval_results
        }, f)
    
    # Generate summary report
    with open(os.path.join(DEFAULT_RESULTS_DIR, "comparison_summary.txt"), 'w') as f:
        f.write("Guitar Chord Recognition Model Comparison\n")
        f.write("=======================================\n\n")
        
        f.write("Accuracy Comparison:\n")
        for model_name, acc in accuracies.items():
            f.write(f"{model_name.upper()} Model: {acc:.4f}\n")
        
        f.write("\nModel Strengths and Weaknesses:\n")
        for model_name in eval_results.keys():
            # Find best and worst performing chords
            f1_scores = class_f1_scores[model_name]
            best_chord = max(f1_scores.items(), key=lambda x: x[1])
            worst_chord = min(f1_scores.items(), key=lambda x: x[1])
            
            f.write(f"\n{model_name.upper()} Model:\n")
            f.write(f"- Best chord: {best_chord[0]} (F1: {best_chord[1]:.4f})\n")
            f.write(f"- Worst chord: {worst_chord[0]} (F1: {worst_chord[1]:.4f})\n")
    
    print(f"Comparison results saved to {DEFAULT_RESULTS_DIR}") 