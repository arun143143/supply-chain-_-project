
# AI-Driven Supply Chain Disruption Predictor&inventory optimization 

Project Statement:

This project aims to develop an advanced AI-powered system that revolutionizes
supply chain management by monitoring global supply chain data, predicting
potential disruptions, and optimizing inventory levels accordingly. By leveraging
LLMs like OpenAI GPT, Meta LLaMA for natural language processing and data
analysis, integrating with Google Sheets, and utilizing Slack or Email for real-time
alerts, the system will provide predictive insights on supply chain risks and automate
inventory adjustments. This comprehensive solution will enable businesses to
proactively manage supply chain disruptions, optimize inventory levels, and maintain
operational continuity in the face of global uncertainties.


## Project Overview
This project is an advanced AI-powered system that revolutionizes supply chain management by:

1.Monitoring Global Supply Chain Data:
Collects and analyzes data related to global supply chains, with a focus on the wheat commodity, to identify trends and disruptions.

2.Predicting Potential Disruptions:
Calculates sentiment analysis and risk factor analysis scores for wheat-related supply chain data using state-of-the-art NLP and ML models (e.g., Hugging Face APIs). These scores are merged, and their average is taken to quantify the overall risk level.

3.Optimizing Inventory Dynamically:
Creates a sample inventory and adjusts stock levels dynamically based on the combined risk and sentiment scores as well as current inventory levels.

4.Sending Real-Time Alerts:
Notifies stakeholders in real time using Discord webhooks, highlighting actionable insights such as stock adjustments and risk levels.

By leveraging large language models like OpenAI GPT, Meta LLaMA, and Hugging Face APIs, this system empowers businesses to proactively manage risks, optimize inventory, and ensure operational continuity amidst global supply chain uncertainties.
## Key Features

1. Global Data Monitoring and Analysis

Collects data from sources like the Event Registry via REST API.
Identifies risks using NLP on news, supplier, and transportation data.

2. Predictive Disruption Modeling

Sentiment analysis with Hugging Face’s BERT model.
Risk scoring and disruption probabilities using Hugging Face’s LargeMini model.

3. ERP Integration and Inventory Optimization

Adjusts inventory dynamically and recommends reorder points.
Provides actionable insights for supply chain adjustments.

4. Real-Time Alerts and Reporting

Sends instant alerts via Discord webhooks and Slack.
Visualizes risks and trends with actionable recommendations.

## Tech Stack

**Languages:** Python

**APIs:** 

**Event Registry API:** For global supply chain data collection.

**Hugging Face APIs:** For NLP tasks like sentiment analysis and risk factor modeling.

**Discord Webhook API:** For real-time alert notifications.

**Libraries and Frameworks:**

**Hugging Face Transformers:** For BERT and LargeMini models.

**pytorch:** PyTorch is an open-source deep learning framework for building and training machine learning models with flexibility and efficiency.

**REST API:** Used to fetch external data.

**Requests:** For API communication.

**Pandas:** Data manipulation and analysis.

**JSON:** Handling API responses.

## Installation

Follow these steps to set up the project on your local machine:

**Prerequisites**

Python 3.10+
Libraries listed in requirements.txt (see below for installation steps).

**API keys for:**

Event Registry (for data collection).

Discord Webhooks (for real-time alerts).

Hugging Face (for BERT and LargeMini models).

1. **Clone the Repository:**

```bash
git clone https://github.com/arun143143/supply-chain-_-project.git
```
2. **Set Up the Environment:**
1. Create a Virtual Environment (recommended):

```bash
python -m venv env
source env/bin/activate    # Linux/Mac
env\Scripts\activate       # Windows
```
2. Install Dependencies:

```bash
pip install -r requirements.txt
```
3. Add API Keys:

Create a .env file in the root directory and add the following:

```bash
EVENT_REGISTRY_API_KEY=your_event_registry_api_key
DISCORD_WEBHOOK_URL=your_discord_webhook_url
HUGGING_FACE_API_KEY=your_hugging_face_api_key
```
3. **Run the Application:**
1.Data Collection:
The first module fetches data using Event Registry's API.

Run:

```bash
python collect_data.py
```
2.Sentiment and Risk Analysis:

Processes data using Hugging Face models:

```bash
python analyze_risk.py
```

3.Inventory Optimization:

Optimizes inventory dynamically based on analysis:

```bash
python optimize_inventory.py
```
4.Real-Time Alerts:
Sends notifications via Discord:

```bash
python send_alerts.py
```

3. **Configure API Keys:**

Obtain your API keys for the Event Registry, Hugging Face, and Discord webhook.

Update the config.py file:
python

```python
EVENT_REGISTRY_API_KEY = "your-event-registry-api-key"
HUGGINGFACE_API_KEY = "your-hugging-face-api-key"
DISCORD_WEBHOOK_URL = "your-discord-webhook-url"
```

4. **Run the System: Execute the main script:**

```bash
python milestone_1.py,
milestone_2.py,
milestone_3.py,
milestone_4.py
```
## Example payload for discard webhook
Example Discord Alert Payload
Here’s an example of the Discord webhook alert payload:

```json
{
    "avatar_url": "https://example.com/image.jpg",
    "username": "Alert System",
    "content": "🚨 **Action Required!** 🚨\n**Month**: January\n**Reason**: Transportation Delays\n**Action**: Increase Stock by 20%",
    "embeds": [{
        "title": "Supply Chain Alert",
        "description": "Details:\n- **Month**: January\n- **Reason**: Transportation Delays\n- **Action**: Increase Stock by 20%"
    }]
}
```
## File Structure

```bash
├── milestone_1
│   ├── data.json                 # Fetched global supply chain data
│   ├── wheat_data_fetching.py    # Script to fetch wheat-related data
│
├── milestone_2
│   ├── analysis_results.csv      # CSV file with analysis output
│   ├── Commodity.txt             # Commodity-related data for analysis
│   ├── sentiment_risk_analysis.py # Script for sentiment and risk analysis
│
├── milestone_3
│   ├── inventory.csv                 # Sample inventory data
│   ├── Inventory.py                   # Inventory optimization logic
│
├── milestone_4
│   ├── alert.py                  # Script to send real-time alerts via Discord
│
├── myenv                         # Virtual environment
├── .env                          # Contains environment variables (API keys)
├── readme.txt
├── supply_chain_analysis.py                     # Documentation
├── requirements.txt              # Python dependencies
├── wheat_data.json               # Processed wheat-related data

```
## Usage

**Data Collection:** Automatically fetches global supply chain data.

**Risk Prediction:** Generates sentiment scores and risk factors for identified events using Hugging Face models.

**Inventory Optimization:** 
Provides actionable insights for stock adjustments.

**Real-Time Alerts:** Sends notifications with details about risks, actions, and reasons.
## License


This project is licensed under the MIT License. See the LICENSE file for more details.

[MIT]
(https://github.com/arun143143/supply-chain-_-project/blob/main/License.txt)




## Contact 
For questions or suggestions, feel free to reach out at:
📧 arunprasathpuni289@gmail.com

