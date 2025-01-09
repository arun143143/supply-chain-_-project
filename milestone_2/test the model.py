import json
import pandas as pd
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    pipeline,
)
from datasets import Dataset

def fine_tune_model(model_name, train_dataset, eval_dataset, output_dir, num_labels):
    """
    Fine-tunes a pre-trained model using the Trainer API.

    Parameters:
        model_name (str): Pre-trained model name (e.g., "bert-base-uncased").
        train_dataset (Dataset): Hugging Face Dataset for training.
        eval_dataset (Dataset): Hugging Face Dataset for evaluation.
        output_dir (str): Directory to save the fine-tuned model.
        num_labels (int): Number of classification labels.

    Returns:
        str: Path to the fine-tuned model directory.
    """
    print("Loading tokenizer and model...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)

    # Define training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        evaluation_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=8,
        num_train_epochs=3,
        save_steps=10_000,
        save_total_limit=2,
    )

    # Define Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
    )

    print("Training the model...")
    trainer.train()

    print(f"Model fine-tuned and saved at {output_dir}")
    return output_dir

def load_datasets(train_file, eval_file):
    """
    Loads training and evaluation datasets.

    Parameters:
        train_file (str): Path to the training dataset JSON file.
        eval_file (str): Path to the evaluation dataset JSON file.

    Returns:
        tuple: Tokenized train and eval datasets.
    """
    print("Loading datasets...")
    train_dataset = Dataset.from_json(train_file)
    eval_dataset = Dataset.from_json(eval_file)

    print("Tokenizing datasets...")
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")  # Replace with your model name
    def tokenize_data(example):
        return tokenizer(example["text"], truncation=True, padding="max_length", max_length=512)

    train_dataset = train_dataset.map(tokenize_data, batched=True)
    eval_dataset = eval_dataset.map(tokenize_data, batched=True)

    return train_dataset, eval_dataset

def perform_sentiment_analysis(df, model_dir):
    """
    Performs sentiment analysis using the fine-tuned sentiment model.

    Parameters:
        df (pd.DataFrame): DataFrame containing 'body' column.
        model_dir (str): Path to the fine-tuned sentiment model.

    Returns:
        pd.DataFrame: DataFrame with 'sentiment_score' column added (1 for positive, 0 for negative).
    """
    print("Loading fine-tuned sentiment analysis model...")
    sentiment_analyzer = pipeline("text-classification", model=model_dir)
    df["sentiment_score"] = df["body"].map(
        lambda x: 1 if sentiment_analyzer(x[:512])[0]["label"] == "LABEL_1" else 0
    )
    return df

def perform_risk_factor_analysis(df, model_dir):
    """
    Performs risk factor classification using the fine-tuned risk factor model.

    Parameters:
        df (pd.DataFrame): DataFrame containing 'body' column.
        model_dir (str): Path to the fine-tuned risk factor model.

    Returns:
        pd.DataFrame: DataFrame with 'risk_score' column added (0 to 2 based on risk classification).
    """
    print("Loading fine-tuned risk factor classification model...")
    risk_classifier = pipeline("text-classification", model=model_dir)
    risk_mapping = {"LABEL_0": 0, "LABEL_1": 1, "LABEL_2": 2}
    df["risk_score"] = df["body"].map(
        lambda x: risk_mapping.get(risk_classifier(x[:512])[0]["label"], 0)
    )
    return df

def main(input_file, train_file, eval_file, sentiment_model_dir, risk_model_dir, output_file):
    # Step 1: Load and fine-tune models
    train_dataset, eval_dataset = load_datasets(train_file, eval_file)
    sentiment_model_dir = fine_tune_model("bert-base-uncased", train_dataset, eval_dataset, sentiment_model_dir, num_labels=2)
    risk_model_dir = fine_tune_model("bert-base-uncased", train_dataset, eval_dataset, risk_model_dir, num_labels=3)

    # Step 2: Load input data
    print("Extracting data...")
    with open(input_file, 'r') as file:
        data = json.load(file)

    articles = data.get("articles", {}).get("results", [])
    df = pd.DataFrame(articles, columns=["body"])
    df.dropna(subset=["body"], inplace=True)

    # Step 3: Perform sentiment and risk factor analysis
    df = perform_sentiment_analysis(df, sentiment_model_dir)
    df = perform_risk_factor_analysis(df, risk_model_dir)

    # Step 4: Save results
    df.to_csv(output_file, index=False)
    print(f"Analysis completed. Results saved to {output_file}")

# Example usage
input_file = r"C:\Users\arunp\OneDrive\Desktop\infosys internship\project\event_registry_data.json"
train_file = r"C:\Users\arunp\OneDrive\Desktop\infosys internship\project\wheat_data.json"
eval_file = r"C:\Users\arunp\OneDrive\Desktop\infosys internship\project\event_registry_data.json"
sentiment_model_dir = "./sentiment_model"
risk_model_dir = "./risk_model"
output_file = r"C:\Users\arunp\OneDrive\Desktop\infosys internship\project\final_analysis_results.csv"

main(input_file, train_file, eval_file, sentiment_model_dir, risk_model_dir, output_file)
