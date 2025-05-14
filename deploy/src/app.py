import streamlit as st
import numpy as np
import tensorflow as tf
from transformers import TFBertForSequenceClassification, BertTokenizer
import matplotlib.pyplot as plt
import seaborn as sns

# Load model and tokenizer
@st.cache_resource
def load_model_and_tokenizer():
    model = TFBertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=3)
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    return model, tokenizer

model, tokenizer = load_model_and_tokenizer()

# Title
st.title("🧠 Hate Speech Detection App")
st.write("Enter text below to check if it's hate speech:")

# User Input
user_input = st.text_area("Input text", height=150)

# Prediction
if st.button("Predict"):
    if user_input.strip() == "":
        st.warning("Please enter some text.")
    else:
        inputs = tokenizer(user_input, return_tensors="tf", padding=True, truncation=True, max_length=128)
        logits = model(**inputs).logits
        probs = tf.nn.softmax(logits, axis=1).numpy()[0]
        label = np.argmax(probs)

        label_map = {0: "Hate Speech", 1: "Offensive", 2: "Neutral"}

        st.subheader("🔍 Result:")
        st.success(f"**{label_map[label]}**")

        st.subheader("📊 Prediction Scores:")
        st.bar_chart(probs)

# Sample Confusion Matrix (static for demo)
st.subheader("🧪 Sample Confusion Matrix")
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
