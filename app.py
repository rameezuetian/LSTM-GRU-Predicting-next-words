import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np 
import pickle 

from tensorflow.keras.preprocessing.sequence import pad_sequences


#  Load the model
model = load_model('next_word_model.h5')


# LOAD THE TOKENIZER
with open('tokenizer.pickle' , 'rb') as handle:
    tokenizer = pickle.load(handle)


# function to predict the next word
def predict_next_word(model , tokenizer , text , max_sequence_length):
    sequence = tokenizer.texts_to_sequences([text])
    if len(sequence) >= max_sequence_length:
        sequence = sequence[0][- (max_sequence_length - 1):]
    padded_sequence = pad_sequences(sequence , maxlen = max_sequence_length - 1 , padding = 'pre')
    predicted_probabilities = model.predict(padded_sequence , verbose = 0)
    predicted_index = np.argmax(predicted_probabilities , axis = -1)[0]
    
    for word , index in tokenizer.word_index.items():
        if index == predicted_index:
            return word
    return ""


# Streamlti app
st.title("Next Word Prediction using LSTM")
input_text = st.text_input("Enter your text here:")
if st.button("Predict Next Word"):
    if input_text.strip() == "":
        st.write("Please enter some text.")
    else:
        max_sequence_length = model.input_shape[1] + 1
        next_word = predict_next_word(model , tokenizer , input_text , max_sequence_length)
        st.write(f"The predicted next word is: **{next_word}**")
        st.write(f"The complete sentence is: **{input_text} {next_word}**")