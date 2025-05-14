import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from transformers import BertTokenizer
import matplotlib.pyplot as plt
import seaborn as sns

# Load model and tokenizer
import tensorflow_hub as hub
from tensorflow.keras.models import load_model

@st.cache_resource
def load():
    model = load_model("models/BERT_HateSpeechDetection.h5", 
                       custom_objects={'KerasLayer': hub.KerasLayer},
                       compile=False)
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    return model, tokenizer


# Title
st.title("🧠 Hate Speech Detection App")
st.write("Enter text below to check if it's hate speech:")

# User input
user_input = st.text_area("Input text", height=150)

# Prediction
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
