import pandas as pd
import re
from transformers import pipeline, BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset
import torch
import matplotlib.pyplot as plt

# Step 1: Load the JSON Data
file_path = r"C:\Users\arunp\OneDrive\Desktop\infosys internship\project\wheat_data.json"
data = pd.read_json(file_path)

# Step 2: Normalize the Nested JSON Structure
articles = pd.json_normalize(data['articles']['results'])

# Step 3: Extract Relevant Columns
extracted_data = articles[['title', 'body', 'date']]

# Step 4: Preprocess the Text
def preprocess_text(text):
    """Clean and preprocess text data."""
    if pd.isna(text):
        return ""
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    text = re.sub(r'\s+', ' ', text).strip()  # Remove extra spaces
    return text

# Apply preprocessing to the `body` column
extracted_data['clean_body'] = extracted_data['body'].apply(preprocess_text)

# Step 5: Load the BERT Model and Tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Step 6: Define Risk Categories for Mapping
risk_labels = ["climate", "geopolitical", "economic", "logistical"]

# Step 7: Classify Each Article
classifier = pipeline("text-classification", model="bert-base-uncased")

def classify_risk(text):
    """Classify text into risk categories using BERT."""
    if not text:
        return "unknown"
    result = classifier(text, truncation=True, max_length=512)  # Limit text length for BERT
    label = risk_labels[int(result[0]['label'].split('_')[-1]) % len(risk_labels)]  # Example mapping
    return label

# Apply Classification
extracted_data['risk_category'] = extracted_data['clean_body'].apply(classify_risk)

# Ensure 'risk_category' is correctly populated before further steps
print(extracted_data.head())  # Check if 'risk_category' is correctly added

# Step 8: Map risk labels to numeric values
label_map = {label: idx for idx, label in enumerate(risk_labels)}
extracted_data['label'] = extracted_data['risk_category'].map(label_map)

# Step 9: Tokenize the Text Data for Efficient Training
def tokenize_function(examples):
    return tokenizer(examples['clean_body'], padding="max_length", truncation=True, max_length=512)

# Convert DataFrame to Hugging Face Dataset
dataset = Dataset.from_pandas(extracted_data[['clean_body', 'label']])
dataset = dataset.map(tokenize_function, batched=True)

# Step 10: Use Hugging Face `train_test_split` to Split the Dataset
dataset = dataset.train_test_split(test_size=0.2, seed=42)

# Get train and validation datasets
train_dataset = dataset['train']
val_dataset = dataset['test']

# Step 11: Define the Model for Fine-Tuning
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=len(risk_labels))

# Step 12: Optimized Training Arguments with Mixed Precision
training_args = TrainingArguments(
    output_dir='./results',          # output directory
    num_train_epochs=3,              # number of training epochs
    per_device_train_batch_size=8,   # batch size for training
    per_device_eval_batch_size=16,   # batch size for evaluation
    warmup_steps=500,                # number of warmup steps for learning rate scheduler
    weight_decay=0.01,               # strength of weight decay
    logging_dir='./logs',            # directory for storing logs
    logging_steps=10,
    evaluation_strategy="epoch",     # Evaluate model after every epoch
    save_strategy="epoch",           # Save model checkpoint after every epoch
    load_best_model_at_end=True,     # Load the best model when training ends
    fp16=True,                       # Use mixed-precision training for faster training
    disable_tqdm=False               # Enable tqdm progress bar for training
)

# Step 13: Initialize the Trainer
trainer = Trainer(
    model=model,                         # the model to be trained
    args=training_args,                  # training arguments
    train_dataset=train_dataset,         # training dataset
    eval_dataset=val_dataset,            # evaluation dataset
)

# Step 14: Fine-Tune the Model
trainer.train()

# Step 15: Save the Fine-Tuned Model and Tokenizer
model.save_pretrained('fine_tuned_bert_wheat_risk')
tokenizer.save_pretrained('fine_tuned_bert_wheat_risk')

# Optional: Evaluate the model after fine-tuning
eval_results = trainer.evaluate()
print("Evaluation Results:", eval_results)

# Step 16: Visualization of Risk Category Distribution
risk_counts = extracted_data['risk_category'].value_counts()
plt.figure(figsize=(8, 5))
risk_counts.plot(kind='bar', color='skyblue')
plt.title("Frequency of Risk Factors in Wheat Supply Chain")
plt.xlabel("Risk Categories")
plt.ylabel("Number of Articles")
plt.xticks(rotation=45)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()

# Step 17: Save the Results to CSV
output_file = "wheat_risk_analysis.csv"
extracted_data.to_csv(output_file, index=False)
print(f"Risk analysis saved to {output_file}.")
