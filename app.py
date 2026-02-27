import streamlit as st
import numpy as np
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load trained model
model = load_model("lstm_model.h5")

# Load tokenizer
with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

# ⚠️ Yaha apna original max_len likho
max_len = 20  

# Reverse word index
index_to_word = {index: word for word, index in tokenizer.word_index.items()}

def predict_next_word(text):
    text = text.lower()
    seq = tokenizer.texts_to_sequences([text])[0]
    seq = pad_sequences([seq], maxlen=max_len, padding='pre')
    pred = model.predict(seq, verbose=0)
    pred_index = np.argmax(pred)
    return index_to_word.get(pred_index, "")

def generate_text(seed_text, n_words):
    for _ in range(n_words):
        next_word = predict_next_word(seed_text)
        if next_word == "":
            break
        seed_text += " " + next_word
    return seed_text

st.title("Next Word Prediction using LSTM")

seed = st.text_input("Enter starting text")

if st.button("Generate"):
    result = generate_text(seed, 15)
    st.success(result)