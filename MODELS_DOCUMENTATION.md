# Models Used in Supply Chain Disruption Predictor Project

## Overview

This AI-driven supply chain disruption predictor leverages several machine learning models to analyze global supply chain data, predict potential disruptions, and optimize inventory levels. This document provides a comprehensive overview of all models used in the project.

## Primary Models

### 1. BERT (Bidirectional Encoder Representations from Transformers)

**Model Name:** `bert-base-uncased`

**Framework:** Hugging Face Transformers

**Purpose:** Primary model for natural language processing tasks including sentiment analysis and risk factor classification.

#### Model Specifications:
- **Architecture:** Transformer-based bidirectional encoder
- **Parameters:** 110M parameters
- **Vocabulary Size:** 30,522 tokens
- **Max Sequence Length:** 512 tokens
- **Hidden Size:** 768
- **Number of Layers:** 12
- **Number of Attention Heads:** 12

#### Applications in the Project:

##### A. Sentiment Analysis
- **File:** `milestone_2/risk_factor_analysis.py`, `milestone_2/test the model.py`
- **Purpose:** Analyzes sentiment of supply chain news articles
- **Implementation:** Fine-tuned BERT model for binary classification (positive/negative sentiment)
- **Output:** Sentiment scores (0 or 1) for each article

```python
# Sentiment Analysis Pipeline
classifier = pipeline("text-classification", model="bert-base-uncased")
sentiment_analyzer = pipeline("text-classification", model=model_dir)
```

##### B. Risk Factor Classification
- **File:** `milestone_2/risk_factor_analysis.py`
- **Purpose:** Classifies articles into risk categories for supply chain analysis
- **Categories:** 
  - Climate risks
  - Geopolitical risks
  - Economic risks
  - Logistical risks
- **Implementation:** Fine-tuned BERT model for multi-class classification
- **Output:** Risk category labels and scores (0-2 scale)

```python
# Risk Factor Classification
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=len(risk_labels))
```

## Model Training and Fine-tuning

### Training Configuration

#### BERT Fine-tuning Parameters:
- **Training Epochs:** 3
- **Batch Size:** 8 (training), 16 (evaluation)
- **Learning Rate:** 2e-5
- **Warmup Steps:** 500
- **Weight Decay:** 0.01
- **Optimization:** Mixed precision training (fp16=True)
- **Max Sequence Length:** 512 tokens

#### Training Process:
1. **Data Preprocessing:** Text cleaning, tokenization using BERT tokenizer
2. **Dataset Split:** 80% training, 20% validation
3. **Fine-tuning:** Supervised learning on wheat supply chain data
4. **Evaluation:** Per-epoch evaluation with best model selection
5. **Model Saving:** Saves both model weights and tokenizer

```python
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=16,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    fp16=True
)
```

## Model Deployment and Usage

### Inference Pipeline

The project uses Hugging Face's pipeline API for model inference:

```python
# Sentiment Analysis Pipeline
sentiment_analyzer = pipeline("text-classification", model=sentiment_model_dir)

# Risk Factor Classification Pipeline  
risk_classifier = pipeline("text-classification", model=risk_model_dir)
```

### Data Flow:
1. **Data Collection:** Articles fetched from Event Registry API
2. **Preprocessing:** Text cleaning and normalization
3. **Tokenization:** BERT tokenizer processes text into model inputs
4. **Inference:** Models predict sentiment and risk factors
5. **Post-processing:** Scores combined for inventory optimization decisions

## Supporting Technologies

### Deep Learning Frameworks
- **PyTorch:** Primary deep learning framework (v2.5.1)
- **TensorFlow:** Secondary framework (v2.18.0)
- **Hugging Face Transformers:** Model library and utilities (v4.48.0)

### Data Processing Libraries
- **Pandas:** Data manipulation and analysis
- **NumPy:** Numerical computing
- **Datasets:** Hugging Face dataset management
- **Scikit-learn:** Additional ML utilities

### NLP Libraries
- **NLTK:** Natural language processing toolkit
- **spaCy:** Advanced NLP processing
- **TextBlob:** Simple text processing

## Model Performance and Outputs

### Sentiment Analysis Model
- **Input:** Article text (max 512 tokens)
- **Output:** Binary classification (LABEL_0: negative, LABEL_1: positive)
- **Score Range:** 0-1 (probability scores)

### Risk Factor Model
- **Input:** Article text (max 512 tokens)  
- **Output:** Multi-class classification (LABEL_0, LABEL_1, LABEL_2)
- **Score Range:** 0-2 (risk level indicators)

## Models Mentioned But Not Implemented

The following models are mentioned in the project documentation but are not currently implemented in the codebase:

### 1. OpenAI GPT
- **Status:** Mentioned in README but not implemented
- **Intended Use:** Advanced natural language processing and analysis

### 2. Meta LLaMA  
- **Status:** Mentioned in README but not implemented
- **Intended Use:** Large language model for enhanced text understanding

### 3. LargeMini Model
- **Status:** Mentioned in README for risk scoring but actual implementation uses BERT
- **Intended Use:** Risk scoring and disruption probabilities

## File Structure for Models

```
├── milestone_2/
│   ├── risk_factor_analysis.py    # BERT fine-tuning and risk classification
│   ├── test the model.py          # Model testing and pipeline setup
│   └── wheat_risk_analysis.csv    # Training/evaluation results
├── fine_tuned_bert_wheat_risk/    # Saved fine-tuned BERT model
├── sentiment_model/               # Saved sentiment analysis model  
└── risk_model/                    # Saved risk factor model
```

## Usage Examples

### Loading and Using Fine-tuned Models

```python
# Load fine-tuned sentiment model
from transformers import pipeline
sentiment_analyzer = pipeline("text-classification", model="./sentiment_model")

# Analyze sentiment
text = "Wheat prices are rising due to supply chain disruptions"
result = sentiment_analyzer(text)
print(f"Sentiment: {result[0]['label']}, Score: {result[0]['score']}")

# Load risk factor model
risk_classifier = pipeline("text-classification", model="./risk_model")

# Classify risk
risk_result = risk_classifier(text)
print(f"Risk Level: {risk_result[0]['label']}")
```

## Future Model Enhancements

Potential improvements and additions to the model architecture:

1. **Integration of LLaMA/GPT models** for enhanced text understanding
2. **Ensemble methods** combining multiple model predictions
3. **Time-series models** for temporal pattern analysis
4. **Multimodal models** incorporating non-text data sources
5. **Specialized domain models** trained on supply chain specific data

## Dependencies and Requirements

Key model-related dependencies from `requirements.txt`:

```
transformers==4.48.0
torch==2.5.1
tensorflow==2.18.0
huggingface-hub==0.27.0
datasets==3.1.0
tokenizers==0.21.0
accelerate==1.1.1
```

## Conclusion

This project primarily leverages BERT-based models for natural language understanding in the supply chain domain. The models are fine-tuned specifically for sentiment analysis and risk factor classification of wheat-related supply chain articles, enabling the system to make informed inventory optimization decisions based on predicted disruptions.