import torch
from datasets import load_dataset
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, Trainer,TrainingArguments 
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

# Load dataset
print("Loading dataset...")
dataset = load_dataset("imdb")

#Load tokenizer
tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased") 

#Tokenization function
def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)

# Apply tokenization (Make sure this runs!)
print("Tokenizing dataset...")
tokenized_datasets = dataset.map(tokenize_function, batched=True)

#Convert to PyTorch tensors
print("Available Columns:", tokenized_datasets["train"].column_names)
tokenized_datasets.set_format(type='torch', columns=["input_ids", "attention_mask", "label"])

#Split into train & test sets
train_dataset = tokenized_datasets["train"]
train_dataset = tokenized_datasets["test"]

# Load model
print("Loading DistilBERT model...")
model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)

# Define training arguments
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch", 
    save_strategy="epoch", 
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8, 
    num_train_epochs=3,
    logging_dir="./logs",
    )

#Initialize Trainer
train_dataset = tokenized_datasets["train"] #Defines training dataset
test_dataset = tokenized_datasets["test"] #Defines test dataset
trainer = Trainer(
    model=model,
    args=training_args, 
    train_dataset=train_dataset, 
    eval_dataset=test_dataset,
    )

# Train model
print("Training model...")
trainer.train()

#Save trained model

print(" Saving model...")
model.saved_pretrained("./models/distilbert_imdb")
tokenizer.save_pretrained("./models/distilbert_imdb")

print("Training complete! Model saved.")

