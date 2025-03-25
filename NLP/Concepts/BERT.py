# Install necessary libraries (if not installed)
!pip install transformers datasets torch scikit-learn

# Import necessary libraries
import torch
import transformers
from transformers import BertTokenizerFast, BertForSequenceClassification
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from datasets import load_dataset
from transformers import AdamW
from sklearn.metrics import accuracy_score

# Load IMDB dataset
dataset = load_dataset("imdb")

# Load BERT tokenizer
tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")

# Hyperparameters
MAX_LENGTH = 256   # Max token length
BATCH_SIZE = 16    # Batch size
LEARNING_RATE = 2e-5  # Learning rate
EPOCHS = 3         # Number of epochs

# Tokenize the dataset
def tokenize_data(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=MAX_LENGTH)

dataset = dataset.map(tokenize_data, batched=True)
dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

# Split into train and validation sets
train_data, val_data = dataset["train"], dataset["test"]

# Create DataLoader
train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_data, batch_size=BATCH_SIZE)

# Load Pretrained BERT model
model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)

# Define optimizer
optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)

# Define training function
def train(model, train_loader, optimizer):
    model.train()
    for batch in train_loader:
        optimizer.zero_grad()
        outputs = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"], labels=batch["label"])
        loss = outputs.loss
        loss.backward()
        optimizer.step()

# Define evaluation function
def evaluate(model, val_loader):
    model.eval()
    predictions, true_labels = [], []
    with torch.no_grad():
        for batch in val_loader:
            outputs = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
            logits = outputs.logits
            predictions.extend(torch.argmax(logits, dim=1).cpu().numpy())
            true_labels.extend(batch["label"].cpu().numpy())
    return accuracy_score(true_labels, predictions)

# Training loop
for epoch in range(EPOCHS):
    train(model, train_loader, optimizer)
    acc = evaluate(model, val_loader)
    print(f"Epoch {epoch+1}/{EPOCHS} - Validation Accuracy: {acc:.4f}")
