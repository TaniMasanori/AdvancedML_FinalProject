# Guitar Chord Recognition

This project implements machine learning models for recognizing guitar chords from audio recordings, specifically focusing on major cowboy chords (E, A, D, G, C, F).

## Overview

The system uses the GuitarSet dataset to train and evaluate three different models:
1. CNN model: A convolutional neural network for spectral analysis
2. BERT-infused model: A transformer-based approach for sequential analysis
3. Naive Bayes model: A traditional machine learning baseline

## Project Structure

```
guitarset_chord_recognition/
├── README.md
├── main.py               # Main script for training and evaluation
├── utils/                
│   ├── data_loader.py    # Data loading and preprocessing utilities
│   ├── preprocessing.py  # Audio preprocessing functions
│   └── visualization.py  # Visualization utilities
├── models/
│   ├── cnn_model.py      # CNN model definition
│   ├── bert_model.py     # BERT model definition
│   └── naive_bayes.py    # Naive Bayes model implementation
├── results/              # Training results and model outputs
└── visualize_results.py  # Script to visualize model performance
```

## Prerequisites

- Python 3.8+
- PyTorch 1.8+
- Librosa
- Numpy
- Matplotlib
- Scikit-learn
- Seaborn

## Installation

```bash
pip install torch torchvision torchaudio
pip install librosa matplotlib numpy scikit-learn seaborn
```

## Dataset

The project uses the GuitarSet dataset, which should be organized as follows:

```
guitarset/
├── audio/            # Contains .wav audio files
└── annotations/      # Contains chord annotation files
```

## Usage

### Training and Evaluation

```bash
cd guitarset_chord_recognition
python main.py --data-dir /path/to/guitarset --annotations-dir /path/to/guitarset/annotations
```

### Visualization

After training, visualize the results:

```bash
python visualize_results.py
```

This will generate:
- A bar chart comparing model accuracies
- Confusion matrices for each model

## Results

### Model Performance
- CNN Model: 89.00% accuracy
- BERT-infused Model: 87.15% accuracy
- Naive Bayes Model: 30.23% accuracy

### Data Distribution
- Total segments processed: 5,952
- A: 1,092 segments
- C: 1,152 segments
- D: 1,144 segments
- E: 880 segments
- F: 920 segments
- G: 764 segments

## Acknowledgments

This project has been enhanced with the assistance of Claude AI by Anthropic.