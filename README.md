# 📧 Spam Email Detection with TensorFlow  

A deep learning project to classify emails as **Spam 🚫** or **Ham ✅** using **TensorFlow + LSTM networks**.  

---

## 📑 Table of Contents
- [Overview](#-overview)  
- [Features](#-features)  
- [Dataset](#-dataset)  
- [Requirements](#-requirements)  
- [Installation](#-installation)  
- [Usage](#-usage)  
- [Model Architecture](#-model-architecture)  
- [Preprocessing](#-preprocessing)  
- [Training](#-training)  
- [Evaluation](#-evaluation)  
- [Results](#-results)  
- [Troubleshooting](#-troubleshooting)  
- [Future Enhancements](#-future-enhancements)  
- [License](#-license)  

---

## 🔎 Overview  
Spam emails clutter inboxes and can be harmful.  
This project uses **LSTM** to learn sequential patterns in text and detect spam with **90%+ accuracy** 🎯.  

---

## ⭐ Features  
- 🤖 LSTM-based deep learning  
- 🧹 Preprocessing: cleaning, stopwords, tokenization  
- ⚖️ Balanced dataset handling  
- 📊 Word clouds & class distribution plots  
- 🛠️ Early stopping + LR scheduler  
- ✅ High performance (90%+ accuracy)  

---

## 📂 Dataset  
- 📌 **5,171 emails**  
- 🏷️ Labels: `spam` / `ham`  
- 📄 Format: CSV → `text`, `label`  

---

## ⚙️ Requirements  
```bash
python 3.7+
tensorflow>=2.8.0
pandas>=1.3.0
numpy>=1.21.0
matplotlib>=3.5.0
seaborn>=0.11.0
scikit-learn>=1.0.0
nltk>=3.6.0
wordcloud>=1.8.0
Download stopwords:

python
Copy code
import nltk
nltk.download('stopwords')
🖥️ Installation
bash
Copy code
git clone https://github.com/yourusername/spam-email-detection.git
cd spam-email-detection

# create virtual env
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate

# install dependencies
pip install -r requirements.txt
🚀 Usage
🔹 Run full pipeline
bash
Copy code
python spam_detection.py
🔹 Quick Example
python
Copy code
from spam_detection import SpamDetector

detector = SpamDetector()
detector.load_data('Emails.csv')
detector.preprocess_data()
detector.train_model()

print(detector.predict("Congrats! You won $1000!"))  # Spam 🚫
🏗️ Model Architecture
scss
Copy code
Input (text seq) → Embedding (32 dim) → LSTM (16 units) → Dense (32, ReLU) → Output (Sigmoid)
Loss: Binary Crossentropy

Optimizer: Adam

Metric: Accuracy

🧹 Preprocessing
✂️ Remove subject, punctuation, lowercase

🛑 Remove stopwords

⚖️ Balance dataset (spam/ham)

🔢 Tokenize & Pad (len=100)

📈 Training
Epochs: 20

Batch: 32

Validation Split: 20%

Callbacks: EarlyStopping ⏹️, ReduceLROnPlateau 📉

✅ Evaluation
📊 Accuracy: ~90%+

📉 Loss curves & history

☁️ Word clouds & spam/ham distribution

🔮 Results
Example Email	Prediction
"Congrats! You won $1000!"	Spam 🚫 (0.95)
"Meeting at 5 PM tomorrow"	Ham ✅ (0.05)

🛠️ Troubleshooting
❌ NLTK error → download stopwords manually

💾 Memory issue → reduce batch size / sequence length

📉 Low accuracy → tune hyperparams, rebalance data

🚀 Future Enhancements
✨ Add Attention mechanism

🌍 Multilingual support

🌐 Deploy as web app

⚡ Real-time filtering

🤝 Ensemble methods

📜 License
MIT License ✔️
