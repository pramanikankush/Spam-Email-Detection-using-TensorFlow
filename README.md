Spam Email Detection Using TensorFlow
A machine learning project that automatically classifies emails as Spam or Ham (Not Spam) using deep learning techniques with TensorFlow and LSTM networks.
Table of Contents

Overview
Features
Dataset
Requirements
Installation
Project Structure
Usage
Model Architecture
Data Preprocessing
Training Process
Evaluation
Results
Contributing
License

Overview
Spam emails are a major concern for email users, cluttering inboxes and potentially containing malicious content. This project implements a deep learning solution using TensorFlow to automatically detect and classify spam emails with high accuracy.
The model uses LSTM (Long Short-Term Memory) networks to understand sequential patterns in email text, making it effective at identifying spam characteristics even in sophisticated spam emails.
Features

Deep Learning Approach: Uses LSTM neural networks for sequential text analysis
Comprehensive Text Preprocessing: Includes stopword removal, punctuation cleaning, and tokenization
Data Balancing: Handles imbalanced datasets through downsampling techniques
Visualization Tools: Word clouds and distribution plots for data analysis
Model Optimization: Implements early stopping and learning rate reduction callbacks
High Accuracy: Achieves reliable spam detection performance

Dataset
The project uses an email dataset containing labeled examples of spam and ham emails. The dataset includes:

Size: 5,171 emails
Columns: 4 columns including text content and labels
Labels: 'spam' and 'ham' (not spam)
Format: CSV file format

Dataset Structure
- text: Email content
- label: Classification (spam/ham)
- Additional metadata columns
Requirements
Python Version

Python 3.7+

Required Libraries
tensorflow>=2.8.0
pandas>=1.3.0
numpy>=1.21.0
matplotlib>=3.5.0
seaborn>=0.11.0
scikit-learn>=1.0.0
nltk>=3.6.0
wordcloud>=1.8.0
NLTK Data
The project requires NLTK stopwords corpus:
pythonimport nltk
nltk.download('stopwords')
Installation

Clone the repository:

bashgit clone https://github.com/yourusername/spam-email-detection.git
cd spam-email-detection

Create a virtual environment (recommended):

bashpython -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

Install dependencies:

bashpip install -r requirements.txt

Download NLTK data:

bashpython -c "import nltk; nltk.download('stopwords')"

Download the dataset:

Place the Emails.csv file in the project directory
Ensure the CSV file has the correct structure with 'text' and 'label' columns

Usage
Basic Usage

Run the complete pipeline:

bashpython spam_detection.py

Step-by-step execution:

pythonimport pandas as pd
import numpy as np
from spam_detection import SpamDetector

# Initialize detector
detector = SpamDetector()

# Load and preprocess data
detector.load_data('Emails.csv')
detector.preprocess_data()

# Train model
detector.train_model()

# Make predictions
result = detector.predict("Congratulations! You've won $1000!")
print(f"Prediction: {'Spam' if result > 0.5 else 'Ham'}")
Custom Configuration
You can customize various parameters:
python# Model parameters
MAX_SEQUENCE_LENGTH = 100
EMBEDDING_DIM = 32
LSTM_UNITS = 16
DENSE_UNITS = 32

# Training parameters
BATCH_SIZE = 32
EPOCHS = 20
VALIDATION_SPLIT = 0.2
Model Architecture
The spam detection model uses the following architecture:
Input Layer (Text Sequences)
    ↓
Embedding Layer (32 dimensions)
    ↓
LSTM Layer (16 units)
    ↓
Dense Layer (32 units, ReLU activation)
    ↓
Output Layer (1 unit, Sigmoid activation)
Architecture Details

Embedding Layer:

Converts word tokens to dense vectors
Dimension: 32
Vocabulary size: Dynamic based on dataset


LSTM Layer:

Captures sequential patterns in text
Units: 16
Handles variable-length sequences


Dense Layers:

Feature extraction: 32 units with ReLU
Output: 1 unit with Sigmoid for binary classification


Compilation:

Loss: Binary Crossentropy
Optimizer: Adam
Metrics: Accuracy



Data Preprocessing
The preprocessing pipeline includes:
1. Text Cleaning

Remove 'Subject' prefixes
Remove punctuation marks
Convert to lowercase

2. Stopword Removal

Remove common English stopwords
Preserve meaningful words for classification

3. Data Balancing

Downsample majority class (Ham)
Create balanced dataset for better training

4. Tokenization and Padding

Convert text to numerical sequences
Pad sequences to uniform length (100 tokens)
Create train/test splits (80/20)

Example Preprocessing Output
Original: "Subject: Free money now!!!"
Cleaned:  "free money"
Tokenized: [45, 123]
Padded:   [45, 123, 0, 0, ..., 0]  # Length 100
Training Process
The model training includes several optimization techniques:
1. Callbacks

EarlyStopping: Prevents overfitting by stopping training when validation accuracy doesn't improve
ReduceLROnPlateau: Reduces learning rate when validation loss plateaus

2. Training Configuration
pythonEarlyStopping(patience=3, monitor='val_accuracy', restore_best_weights=True)
ReduceLROnPlateau(patience=2, monitor='val_loss', factor=0.5, verbose=0)
3. Training Parameters

Epochs: 20 (with early stopping)
Batch Size: 32
Validation Split: 20%

Evaluation
The model performance is evaluated using:

Accuracy: Overall classification accuracy
Validation Loss: Model loss on validation data
Training History: Loss and accuracy curves over epochs

Visualization Tools

Class Distribution: Shows spam vs ham email counts
Word Clouds: Visual representation of frequent words in each class
Training History: Plots of loss and accuracy during training

Results
Expected model performance:

Training Accuracy: ~95%+
Validation Accuracy: ~90%+
Training Time: 2-5 minutes (depending on hardware)

Sample Predictions
python# Spam examples
"Congratulations! You won $1000!" → Spam (0.95)
"Click here for free money!"      → Spam (0.89)

# Ham examples  
"Meeting scheduled for tomorrow"   → Ham (0.05)
"Thanks for your email"           → Ham (0.12)
API Reference
Main Classes
pythonclass SpamDetector:
    def load_data(self, filepath)
    def preprocess_data(self)
    def train_model(self, epochs=20)
    def predict(self, text)
    def save_model(self, filepath)
    def load_model(self, filepath)
Utility Functions
pythondef remove_punctuations(text)
def remove_stopwords(text)  
def plot_word_cloud(data, typ)
def plot_class_distribution(data)
Contributing

Fork the repository
Create a feature branch (git checkout -b feature/AmazingFeature)
Commit your changes (git commit -m 'Add some AmazingFeature')
Push to the branch (git push origin feature/AmazingFeature)
Open a Pull Request

Development Setup

Install development dependencies:

bashpip install -r requirements-dev.txt

Run tests:

bashpython -m pytest tests/

Check code style:

bashflake8 spam_detection.py
Troubleshooting
Common Issues

NLTK Download Error:

bashpython -c "import ssl; ssl._create_default_https_context = ssl._create_unverified_context; import nltk; nltk.download('stopwords')"

Memory Issues:

Reduce batch size to 16 or 8
Reduce maximum sequence length


Low Accuracy:

Check data quality and balance
Increase model complexity
Tune hyperparameters



Performance Tips

Use GPU acceleration for faster training
Implement data generators for large datasets
Consider using pre-trained embeddings (Word2Vec, GloVe)

Future Enhancements

 Implement attention mechanisms
 Add support for multilingual emails
 Deploy model as web service
 Add real-time email filtering
 Implement ensemble methods
 Add explainability features

License
This project is licensed under the MIT License - see the LICENSE file for details.
Acknowledgments

TensorFlow team for the deep learning framework
NLTK contributors for natural language processing tools
Dataset providers for email classification data
Open source community for various Python libraries
