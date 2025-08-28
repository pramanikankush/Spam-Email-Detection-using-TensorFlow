📧 Spam Email Detection using TensorFlow

A deep learning project to automatically classify emails as Spam or Ham (Not Spam) using TensorFlow and LSTM networks.

This project demonstrates data preprocessing, balancing imbalanced classes, text cleaning, tokenization, and model training with callbacks for optimal performance.

🚀 Features

Preprocessing pipeline for text cleaning (stopwords, punctuation removal).

Class balancing using downsampling.

WordCloud visualization for insights into email text.

Tokenizer & Padding to convert text into numerical sequences.

Deep Learning Model: Embedding + LSTM + Dense Layers.

Training with EarlyStopping & ReduceLROnPlateau.

Achieves high accuracy on email spam classification.

📂 Dataset

The dataset contains 5171 rows and 4 columns of labeled emails.

label: Email type (spam / ham)

text: The email content

Other metadata columns

Download Dataset: Emails.csv
 (Add your dataset link here)

🛠️ Tech Stack

Python 3.x

TensorFlow / Keras

NLTK (Stopwords removal)

Pandas, NumPy, Matplotlib, Seaborn

WordCloud for visualization

Scikit-learn for data splitting

📊 Workflow
1️⃣ Import Libraries

Load essential libraries for preprocessing, visualization, and deep learning.

2️⃣ Load Dataset
data = pd.read_csv('Emails.csv')
print(data.shape)  # (5171, 4)

3️⃣ Handle Class Imbalance

Downsample the Ham class to balance with Spam emails.

4️⃣ Preprocess Text

Remove Subject

Remove punctuations

Remove stopwords

Convert text to lowercase

5️⃣ Visualize Data

Label distribution using Seaborn

Frequent words using WordCloud

6️⃣ Tokenization & Padding

Convert emails to sequences of integers and pad them to equal length.

7️⃣ Model Architecture
model = tf.keras.models.Sequential([
    tf.keras.layers.Embedding(input_dim=len(tokenizer.word_index)+1, output_dim=32, input_length=100),
    tf.keras.layers.LSTM(16),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

8️⃣ Training

Use EarlyStopping & ReduceLROnPlateau to optimize learning.

history = model.fit(
    train_sequences, train_Y,
    validation_data=(test_sequences, test_Y),
    epochs=20,
    batch_size=32,
    callbacks=[lr, es]
)

📈 Model Summary

Embedding Layer → Word vector representation

LSTM Layer → Sequence learning

Dense Layers → Feature extraction

Sigmoid Output Layer → Binary classification (Spam/Ham)

Parameters: ~1.28M trainable

📉 Results

Accuracy: ~90%+ (depending on dataset split & preprocessing)

Balanced performance on Spam & Ham emails

🔮 Future Improvements

Use Bi-directional LSTM/GRU for better sequence capture

Try pre-trained embeddings (GloVe, Word2Vec)

Experiment with transformer-based models (BERT, DistilBERT)

Deploy model as a Flask/Streamlit Web App

▶️ How to Run

Clone the repo

git clone https://github.com/yourusername/spam-email-detection.git
cd spam-email-detection


Install requirements

pip install -r requirements.txt


Run the script

python spam_detector.py


Train the model & check results

📌 Requirements

requirements.txt should include:

numpy
pandas
matplotlib
seaborn
nltk
wordcloud
scikit-learn
tensorflow
keras

🙌 Contribution

Pull requests are welcome! For major changes, open an issue first to discuss what you’d like to add.

📜 License

This project is licensed under the MIT License.
