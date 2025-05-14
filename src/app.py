import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from transformers import BertTokenizer
import matplotlib.pyplot as plt
import seaborn as sns

# Load model and tokenizer
import tensorflow as tf
import tensorflow_hub as hub
from transformers import BertTokenizer

@st.cache_resource
def load():
    # โหลด BERT preprocessing และ encoder จาก TF Hub
    preprocess = hub.KerasLayer("https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3")
    encoder = hub.KerasLayer("https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/4")

    # สร้างโมเดลใหม่ (ย่อ)
    text_input = tf.keras.layers.Input(shape=(), dtype=tf.string)
    encoder_inputs = preprocess(text_input)
    outputs = encoder(encoder_inputs)['pooled_output']
    outputs = tf.keras.layers.Dense(3, activation='softmax')(outputs)
    model = tf.keras.Model(inputs=text_input, outputs=outputs)

    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    return model, tokenizer



# Title
st.title("🧠 Hate Speech Detection App")
st.write("Enter text below to check if it's hate speech:")

# User input
user_input = st.text_area("Input text", height=150)

# Prediction
model, tokenizer = load()
if st.button("Predict"):
    if user_input.strip() == "":
        st.warning("Please enter some text.")
    else:
        inputs = tokenizer(user_input, return_tensors="tf", padding=True, truncation=True)
        prediction = model.predict(inputs["input_ids"])[0]
        label = np.argmax(prediction)

        label_map = {0: "Hate Speech", 1: "Offensive", 2: "Neutral"}
        st.subheader("🔍 Result:")
        st.write(f"**{label_map[label]}**")

        # Optional: show probability
        st.write("Prediction scores:")
        st.bar_chart(prediction)

# Visualization (static example)
st.subheader("📊 Model Performance Example (Static)")
conf_matrix = np.array([[90, 5, 5],
                        [7, 85, 8],
                        [3, 6, 91]])

fig, ax = plt.subplots()
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Hate", "Offensive", "Neutral"],
            yticklabels=["Hate", "Offensive", "Neutral"])
ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")
st.pyplot(fig)
