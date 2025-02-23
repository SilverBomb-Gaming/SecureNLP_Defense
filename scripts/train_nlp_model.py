from datasets import load_dataset
from transformers import DistilBertTokenizer

# Load IMDB dataset
print("Loading dataset...")
dataset = load_dataset("imdb")

# Load DistilBERT tokenizer
tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")

#Tokenization function
def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)

# Apply tokenization
print("Tokenizing dataset...")
tokenized_datasets = dataset.map(tokenize_function, batched=True)

# Print sample
print("Tokenization Complete. Sample:")
print(tokenized_datasets["train"][0])

from transformers import DistilBertForSequenceClassification

# Load DistilBERT model
print("Loading DistilBERT model...")
model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased")

# Save the trained model
print("Saving trained model...")
model.save_pretrained("./models/distilbert_imdb")
print("Model saved successfully!")

# Save the trained model
print("Saving trained model...")
model.save_pretrained("./models/distilbert_imdb")
print("Model saved successfully!")
