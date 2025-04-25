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

This project uses the GuitarSet dataset, a collection of high-quality guitar recordings with rich annotations including chord labels, pitch contours, and playing style information.

### GuitarSet Dataset Information
- **360 excerpts** (approximately 30 seconds each)
- Recorded by **6 different players**
- Includes **comping and soloing versions**
- Covers **5 styles**: Rock, Singer-Songwriter, Bossa Nova, Jazz, and Funk
- Based on **3 chord progressions**: 12 Bar Blues, Autumn Leaves, and Pachelbel Canon
- Recorded at **2 different tempi**: slow and fast
- Audio recorded using a **hexaphonic pickup** (individual string signals) and reference microphone
- Comprehensive **annotations in JAMS format** including pitch contours, note information, beat positions, and chord labels

### Dataset Organization

```
guitarset/
├── audio/            # Contains .wav audio files
├── audio_pickup/     # Individual string recordings
├── audio_mic/        # Microphone recordings
├── annotations/      # Contains chord annotation files
├── test_data/        # Test dataset split
└── training/         # Training dataset split
```

### Citation

If you use GuitarSet for academic purposes, please cite the following publication:

```
Q. Xi, R. Bittner, J. Pauwels, X. Ye, and J. P. Bello, "Guitarset: A Dataset for Guitar Transcription", 
in 19th International Society for Music Information Retrieval Conference, Paris, France, Sept. 2018.
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

### Claude AI
This project has been enhanced with the assistance of Claude 3.7 Sonnet, an AI assistant developed by Anthropic. Claude contributed to:

- Code optimization and refactoring
- Debugging complex model architectures
- Data preprocessing workflow improvements
- Implementation of visualization tools
- Documentation of the codebase
- Research on optimal hyperparameters

### GuitarSet Dataset
Special thanks to the creators of the GuitarSet dataset:
- Qingyang Xi (NYU's Music and Audio Research Lab)
- Rachel Bittner (NYU's Music and Audio Research Lab)
- Johan Pauwels (Center for Digital Music at Queen Mary University of London)
- Xuzhou Ye (NYU's Music and Audio Research Lab)
- Juan Pablo Bello (NYU's Music and Audio Research Lab)

## License
This project is licensed under the MIT License - see the LICENSE file for details.