# 🧠 Next Word Prediction using RNN & LSTM

An end-to-end NLP project that implements a Next Word Prediction system using Simple RNN and LSTM neural networks.

The models were trained on a dataset of **3,038 quotes**, where custom preprocessing, tokenization, and sequence generation were applied to build a language modeling pipeline.

The final LSTM model is deployed as an interactive web application using Streamlit for real-time text generation.

---

## 🚀 Live Demo
https://next-word-predictor-yash-pratap-rai.streamlit.app/

---

## 📌 Project Overview

This project demonstrates:

- Text preprocessing (lowercasing, punctuation removal)
- Tokenization using Keras Tokenizer
- Sequence generation for language modeling
- Padding for uniform input length
- Training using SimpleRNN and LSTM architectures
- Model saving and loading
- Deployment using Streamlit

---

## 🏗️ Dataset

- Total Quotes: **3,038**
- Domain: Motivational and general quotes
- Preprocessing Steps:
  - Lowercasing
  - Punctuation removal
  - Tokenization
  - Sequence generation
  - Padding

---

## 🤖 Model Architecture

### 🔹 Embedding Layer
Converts words into dense vector representations.

### 🔹 Recurrent Layer
- SimpleRNN (baseline comparison)
- LSTM (final deployed model)

### 🔹 Dense Output Layer
Softmax activation to predict next word from vocabulary.

---

## 🛠️ Tech Stack

- Python
- TensorFlow / Keras
- NumPy
- Streamlit
## 🛠️ Deep Learning Architecture
- SimpleRNN
- LSTM
---

## 🎯 Key Features

- Predicts next word based on input text
- Generates full sentence continuation
- Interactive web interface
- Cloud deployment ready
- Comparison between SimpleRNN and LSTM

---

## 📦 Project Structure
next-word-predictor/
│
├── app.py
├── lstm_model.h5
├── tokenizer.pkl
├── requirements.txt
└── README.md

---

## 💡 How It Works

1. User enters seed text.
2. Text is tokenized and padded.
3. Model predicts probability distribution for next word.
4. Highest probability word is selected.
5. Process repeats to generate full sentence.

---

## 📈 Future Improvements

- Temperature-based sampling
- Top-k sampling
- Larger dataset training
- Transformer-based model upgrade

---

## 👨‍💻 Author

Yash Pratap Rai  
Aspiring Data Scientist | ML Enthusiast
