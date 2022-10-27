import streamlit as st

import keras
from keras_preprocessing import sequence
from keras.datasets import imdb
(x_train, y_train),(x_test, y_test) = imdb.load_data(num_words=5000)

x_train = sequence.pad_sequences(x_train, 500)
x_test = sequence.pad_sequences(x_test, 500)

model = keras.models.load_model('imdb.h5')

word_to_id = imdb.get_word_index()
word_to_id = {i:(j+3) for i,j in word_to_id.items()}

# kullanıcının girdiği yorumun tahmini

def user_input_processing(review):
    vec = []
    for word in review.split(" "):
        if word[-1] == ".":
            word = word[:-1]
        vec.append(word_to_id[str.lower(word)])
    vec_padded = sequence.pad_sequences([vec], 500)
    print(review, model.predict(vec_padded))
    if model.predict(vec_padded) > 0.65:
        return "Olumlu yorum."
    if model.predict(vec_padded) < 0.65 and model.predict(vec_padded) > 0.45:
        return "Ortalama yorum."
    if model.predict(vec_padded) < 0.45:
        return "Olumsuz youm."

review = st.text_area('Yorum:')
button = st.button("Onayla.")

if button:
    st.write(user_input_processing(review))