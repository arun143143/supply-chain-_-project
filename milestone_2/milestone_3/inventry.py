import pandas as pd
import os
from datetime import datetime

class SupplyChainAnalyzer:
    def __init__(self):
        self.high_risk_threshold = 0.7
        self.low_sentiment_threshold = 0.4
        
    def load_and_process_data(self, file_path):
        """Load and process the supply chain data from CSV."""
        try:
            # Read the CSV file
            data = pd.read_csv(file_path)
            
            # Clean and standardize column names
            data.columns = data.columns.str.strip().str.lower()
            
            # Define possible column names
            required_columns = ['risk_label', 'risk_score', 'sentiment_score', 'date', 'title', 'sentiment_label']
            
            # Check if required columns are present
            missing_columns = [col for col in required_columns if col not in data.columns]
            if missing_columns:
                raise ValueError(f"Missing required columns: {missing_columns}")
            
            # Process the data
            processed_data = self._process_data(data)
            
            return processed_data
            
        except Exception as e:
            raise Exception(f"Error processing file: {str(e)}")
    
    def _process_data(self, data):
        """Process the data and add recommendations."""
        # Convert date to datetime
        data['date'] = pd.to_datetime(data['date'], errors='coerce')
        
        # Check for any failed date conversions
        if data['date'].isna().any():
            # Drop rows with invalid dates
            data = data.dropna(subset=['date'])
        
        data['month'] = data['date'].dt.month_name()
        
        # Convert scores to numeric if they aren't already
        data['risk_score'] = pd.to_numeric(data['risk_score'], errors='coerce')
        data['sentiment_score'] = pd.to_numeric(data['sentiment_score'], errors='coerce')
        
        # Generate actions and reasons based on scores
        data['action'], data['reason'] = zip(*data.apply(self._get_action_and_reason, axis=1))
        
        # Select only the columns to be saved: Month, Reason, and Action
        data = data[['month', 'reason', 'action']]
        
        # Sort by month
        data = data.sort_values('month')
        
        return data
    
    def _get_action_and_reason(self, row):
        """Determine action and reason based on scores."""
        risk_score = float(row['risk_score'])
        sentiment_score = float(row['sentiment_score'])
        title = str(row['title']).lower()
        
        if risk_score > self.high_risk_threshold:
            if sentiment_score < self.low_sentiment_threshold:
                action = "ALERT"
                reason = self._generate_alert_reason(title)
            else:
                action = "MONITOR"
                reason = self._generate_monitor_reason(title)
        elif sentiment_score < self.low_sentiment_threshold:
            action = "CAUTION"
            reason = self._generate_caution_reason(title)
        else:
            action = "STABLE"
            reason = "No immediate action needed."
            
        return action, reason
    
    def _generate_alert_reason(self, title):
        """Generate detailed alert reason based on title keywords."""
        if 'shortage' in title or 'demand' in title:
            return "High risk and negative sentiment - Immediately increase stock levels!"
        elif 'delay' in title or 'disruption' in title:
            return "High risk and negative sentiment - Prioritize alternative suppliers!"
        else:
            return "High risk and negative sentiment - Prioritize stock!"
    
    def _generate_monitor_reason(self, title):
        """Generate monitoring reason based on title keywords."""
        if 'price' in title:
            return "High risk - Monitor prices and consider bulk purchasing."
        else:
            return "High risk - Keep an eye on supply chain."
    
    def _generate_caution_reason(self, title):
        """Generate caution reason based on title keywords."""
        if 'supply' in title:
            return "Negative sentiment - Consider increasing safety stock."
        else:
            return "Negative sentiment - Consider buying stock."

def main():
    analyzer = SupplyChainAnalyzer()
    
    while True:
        file_path = input("\nEnter the path to your CSV file (or 'q' to quit): ")
        
        if file_path.lower() == 'q':
            break
            
        if not os.path.exists(file_path):
            print("File not found. Please check the path and try again.")
            continue
            
        if not file_path.endswith('.csv'):
            print("Please provide a CSV file.")
            continue
            
        try:
            # Process the file
            results = analyzer.load_and_process_data(file_path)
            
            # Save processed results with only the columns you need
            output_file = f"supply_chain_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            results.to_csv(output_file, index=False)
            print(f"\nAnalysis saved to: {output_file}")
            
        except Exception as e:
            print(f"Error: {str(e)}")

if __name__ == "__main__":
    print("Supply Chain Disruption Analysis System")
    print("======================================")
    main()
