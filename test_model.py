from transformers import DistilBertForSequenceClassification, DistilBertTokenizer
import torch

# Load the trained model and tokenizer
model_path = "./models/distilbert_imdb"
print("Loading model for inference...")
model = DistilBertForSequenceClassification.from_pretrained(model_path)
tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")

# Sample text input
text = "This movie was absolutely fantastic! The acting was great, and the storyline was engaging."

# Tokenize input
inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)

# Run inference
model.eval()
with torch.no_grad():
    output = model(**inputs)

# Interpret result
prediction = torch.argmax(output.logits, dim=1).item()
sentiment = "Positive" if prediction == 1 else "Negative"

print(f"Predicted Sentiment: {sentiment}")
