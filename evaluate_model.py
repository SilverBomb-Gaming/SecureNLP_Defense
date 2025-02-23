from datasets import load_dataset
from transformers import DistilBertForSequenceClassification, DistilBertTokenizer
import torch

# Load test dataset
print("Loading IMDB test dataset...")
dataset = load_dataset("imdb", split="test")

# Load trained model and tokenizer
model_path = "./models/distilbert_imdb"
model = DistilBertForSequenceClassification.from_pretrained(model_path)
tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")

# Select a batch of samples to evaluate
num_samples = 100  # Adjust as needed
correct = 0

print(f"Evaluating on {num_samples} samples...")

for i in range(num_samples):
    text = dataset[i]["text"]
    true_label = dataset[i]["label"]  # 1 = Positive, 0 = Negative

    # Tokenize input
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)

    # Run inference
    model.eval()
    with torch.no_grad():
        output = model(**inputs)

    # Interpret result
    prediction = torch.argmax(output.logits, dim=1).item()
    
    if prediction == true_label:
        correct += 1

accuracy = correct / num_samples * 100
print(f"Model Accuracy on {num_samples} samples: {accuracy:.2f}%")
