import random
import torch
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split

# Load data from labeled_data.csv
data = pd.read_csv("labeled_data.csv")
texts = data["tweet"].tolist()
labels = data["class"].tolist()  # Use the "class" column for labels

# Split the dataset into training and testing sets
random.seed(42)
X_train, X_test, y_train, y_test = train_test_split(texts, labels, test_size=0.2, random_state=42)

model_name = "bert-base-uncased"
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForSequenceClassification.from_pretrained(model_name, num_labels=3)  # Assuming 3 classes: hate_speech, offensive_language, neither

max_seq_length = 128
X_train_tokens = tokenizer(X_train, padding=True, truncation=True, max_length=max_seq_length, return_tensors="pt", return_attention_mask=True)
X_test_tokens = tokenizer(X_test, padding=True, truncation=True, max_length=max_seq_length, return_tensors="pt", return_attention_mask=True)

# Convert labels to tensors
label2id = {"hate_speech": 0, "offensive_language": 1, "neither": 2}
y_train_ids = torch.tensor([label2id[label] for label in y_train])
y_test_ids = torch.tensor([label2id[label] for label in y_test])

batch_size = 32
train_dataset = TensorDataset(X_train_tokens.input_ids, X_train_tokens.attention_mask, y_train_ids)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_dataset = TensorDataset(X_test_tokens.input_ids, X_test_tokens.attention_mask, y_test_ids)
test_loader = DataLoader(test_dataset, batch_size=batch_size)

optimizer = AdamW(model.parameters(), lr=2e-5)
num_epochs = 3

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

for epoch in range(num_epochs):
    model.train()
    for batch in train_loader:
        input_ids, attention_mask, labels = batch
        input_ids, attention_mask, labels = input_ids.to(device), attention_mask.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()

# Evaluation
model.eval()
predictions = []
true_labels = []
with torch.no_grad():
    for batch in test_loader:
        input_ids, attention_mask, labels = batch
        input_ids, attention_mask, labels = input_ids.to(device), attention_mask.to(device), labels.to(device)
        
        outputs = model(input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        batch_predictions = torch.argmax(logits, dim=1).cpu().numpy()
        predictions.extend(batch_predictions)
        true_labels.extend(labels.cpu().numpy())

accuracy = accuracy_score(true_labels, predictions)
print(f"Accuracy: {accuracy:.2f}")
print(classification_report(true_labels, predictions, target_names=["hate_speech", "offensive_language", "neither"]))
