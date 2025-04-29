```mermaid
graph TD
    %% Main Data Flow
    AudioFiles[Raw Audio Files] --> Preprocessing[Preprocessing]
    Preprocessing --> |Create MEL Spectrograms| ProcessedData[Processed Data]
    ProcessedData --> DataLoaders[Create Data Loaders]
    
    %% CNN and BERT Flow
    DataLoaders --> |Training Data| ModelTraining[Model Training]
    ModelTraining --> CNNModel[CNN Model]
    ModelTraining --> BERTModel[BERT-infused Model]
    
    %% Naive Bayes Flow with Frequency Analysis
    AudioSignals[Raw Audio Signals] --> FrequencyExtraction[FFT Feature Extraction]
    FrequencyExtraction --> |Musical Note Frequencies| KeyFrequencies[12 Key Frequency Features]
    KeyFrequencies --> ThresholdAnalysis[Threshold-based Feature Analysis]
    ThresholdAnalysis --> NaiveBayesTraining[Train GaussianNB Model]
    NaiveBayesTraining --> NaiveBayesModel[Naive Bayes Model]
    
    %% Evaluation Flow
    DataLoaders --> |Validation Data| ModelEvaluation[Model Evaluation]
    CNNModel --> ModelEvaluation
    BERTModel --> ModelEvaluation
    NaiveBayesModel --> ModelEvaluation
    ModelEvaluation --> ConfusionMatrix[Confusion Matrices]
    ModelEvaluation --> AccuracyComparison[Accuracy Comparison]
    ModelEvaluation --> F1Scores[F1 Score Comparison]
    
    %% CNN & BERT Prediction Flow
    NewAudio[New Audio File] --> MelSpectrogramGen[Generate MEL Spectrogram]
    MelSpectrogramGen --> ModelsPrediction[Model Prediction]
    CNNModel --> ModelsPrediction
    BERTModel --> ModelsPrediction
    
    %% Naive Bayes Prediction Flow
    NewAudio --> FFTExtraction[FFT Extraction]
    FFTExtraction --> |12 Musical Frequencies| FrequencyFeatures[Frequency Features]
    FrequencyFeatures --> |Feature Magnitude Ratios| NaiveBayesPrediction[Naive Bayes Prediction]
    NaiveBayesModel --> NaiveBayesPrediction
    
    %% Combined Results
    ModelsPrediction --> PredictedChord[Predicted Chord]
    NaiveBayesPrediction --> PredictedChord
    
    %% Visualization Flow
    FFTExtraction --> SpectrumVisualization[Spectrum Visualization]
    FrequencyFeatures --> FeatureVisualization[Feature Visualization]
    SpectrumVisualization --> VisualAnalysis[Visual Analysis]
    FeatureVisualization --> VisualAnalysis
    
    %% Default Configuration
    DefaultConfig[Default Configuration] -->|Cowboy Chords Only| ChordProcessing[Chord Processing]
    ChordProcessing --> Preprocessing
    
    %% Command Options
    Options[Command Options] --> |--data-dir| DataSource[Data Source]
    Options --> |--all-chords| ChordFilter[Use All Chords]
    Options --> |--gpu| GPUSelection[GPU Device Selection]
    Options --> |--use-cpu| CPUForce[Force CPU Usage]
    Options --> |--epochs| TrainingEpochs[Training Epochs]
    
    %% Special Visualization Options
    VisualizationOption[Spectrum Visualization] --> SpectrumVisualization
    
    %% Styling
    classDef newFeature fill:#f96,stroke:#333,stroke-width:2px
    classDef defaultConfig fill:#f9f,stroke:#333,stroke-width:2px
    class FrequencyExtraction,KeyFrequencies,ThresholdAnalysis,FFTExtraction,FrequencyFeatures,SpectrumVisualization,FeatureVisualization newFeature
    class DefaultConfig defaultConfig
```

## GuitarSet Chord Recognition Updated Workflow

### Key Components

1. **Preprocessing & Data Loading**:
   - Process raw audio files into MEL spectrograms
   - Create data loaders for training and validation

2. **Model Training**:
   - **CNN Model**: Processes MEL spectrograms directly
   - **BERT Model**: Uses self-attention mechanism after CNN feature extraction
   - **Naive Bayes Model (Updated)**: 
     - Uses FFT to analyze frequency content of audio
     - Extracts 12 features based on musical note frequencies (C through B)
     - Applies threshold-based feature extraction (signal/noise ratio)
     - Trains a Gaussian Naive Bayes classifier on these musical features

3. **Model Evaluation**:
   - Generate confusion matrices for all models
   - Compare accuracy and F1 scores
   - Analyze model strengths and weaknesses

4. **Prediction**:
   - For CNN & BERT: Convert audio to MEL spectrogram
   - For Naive Bayes: Extract frequency features from audio signal
   - Combine predictions from all models

5. **Visualization**:
   - Analyze frequency spectrum with key musical notes marked
   - Visualize feature extraction process
   - Compare feature importance for chord recognition

### Updated Naive Bayes Process

The key improvement is the new feature extraction process:

1. **FFT Analysis**: Apply Fast Fourier Transform to audio signal
2. **Key Frequency Detection**: Extract magnitude at 12 musical note frequencies
3. **Threshold Analysis**: Compare magnitude to noise threshold (15x mean)
4. **Feature Generation**: Create feature vector of amplitude ratios
5. **Classification**: Train Gaussian Naive Bayes on these musical features

This approach better aligns with musical theory by focusing directly on the frequencies that define musical notes, which should improve chord recognition accuracy.
