# What Models Are Used in This Project?

## Quick Answer

This supply chain disruption predictor project primarily uses **BERT (Bidirectional Encoder Representations from Transformers)** models for natural language processing tasks.

## Models Used

### 1. Primary Model: BERT
- **Specific Model:** `bert-base-uncased` from Hugging Face
- **Framework:** Hugging Face Transformers + PyTorch
- **Use Cases:**
  - **Sentiment Analysis** of supply chain news articles
  - **Risk Factor Classification** (climate, geopolitical, economic, logistical risks)

### 2. Model Applications
| Application | Model | Purpose | Output |
|-------------|-------|---------|---------|
| Sentiment Analysis | Fine-tuned BERT | Analyze positive/negative sentiment in news | 0-1 sentiment scores |
| Risk Classification | Fine-tuned BERT | Classify articles by risk type | Risk category labels (0-2) |

### 3. Supporting Technologies
- **PyTorch** (v2.5.1) - Deep learning framework
- **Hugging Face Transformers** (v4.48.0) - Model library
- **TensorFlow** (v2.18.0) - Additional framework support

## Model Implementation Files
- `milestone_2/risk_factor_analysis.py` - Main BERT fine-tuning and risk analysis
- `milestone_2/test the model.py` - Model testing and pipeline setup

## Models Mentioned But NOT Implemented
- OpenAI GPT (mentioned in documentation only)
- Meta LLaMA (mentioned in documentation only)
- LargeMini model (mentioned but BERT is used instead)

## Summary
**The project uses BERT as the core AI model, fine-tuned specifically for supply chain text analysis to predict disruptions and optimize inventory.**