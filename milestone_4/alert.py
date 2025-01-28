import os
import csv
import time
import requests
from dotenv import load_dotenv


class SupplyChainAlert:
    def __init__(self):
        # Load environment variables from .env file
        load_dotenv()
        self.discard_webhook_url = os.getenv("DISCORD_WEBHOOK_URL")
        if not self.discard_webhook_url:
            raise ValueError("Webhook URL is missing. Please set it in the .env file.")

    def send_discard_alert(self, row):
        """Send alert to Discord webhook."""
        # Validate content
        content_message = f"🚨 **Action Required!** 🚨\n**Month**: {row['month']}\n**Reason**: {row['reason']}\n**Action**: {row['action']}"
        if len(content_message) > 2000:  # Discord message content limit
            content_message = content_message[:2000] + "..."

        # Prepare the payload
        payload = {
            "avatar_url": "https://www.science.org/do/10.1126/science.abi5787/abs/main_puppies_1280p.jpg",
            "username": "Alert System",
            "content": content_message,
            "embeds": [{
                "title": "Supply Chain Alert",
                "description": f"Details:\n- **Month**: {row['month']}\n- **Reason**: {row['reason']}\n- **Action**: {row['action']}"
            }]
        }

        try:
            response = requests.post(self.discard_webhook_url, json=payload)

            if response.status_code == 204:
                print(f"Successfully sent alert for month: {row['month']}.")
            elif response.status_code == 429:  # Handle rate limiting
                retry_after = response.json().get("retry_after", 1)
                print(f"Rate limited. Retrying after {retry_after} seconds...")
                time.sleep(retry_after)
                self.send_discard_alert(row)  # Retry the same request
            else:
                print(f"Failed to send Discord alert for month: {row['month']}. Error: {response.status_code} - {response.text}")
        except Exception as e:
            print(f"Error sending Discord alert: {str(e)}")

    def process_csv_and_send_alerts(self, file_path):
        """Process a CSV file and send alerts to Discord based on its contents."""
        try:
            # Open and read the CSV file
            with open(file_path, mode="r") as file:
                csv_reader = csv.DictReader(file)

                # Iterate through each row in the CSV file
                for row in csv_reader:
                    if all(key in row for key in ["month", "reason", "action"]):
                        self.send_discard_alert(row)
                    else:
                        print(f"Skipping row due to missing data: {row}")
        except FileNotFoundError:
            print(f"File not found: {file_path}")
        except Exception as e:
            print(f"An error occurred while processing the CSV file: {str(e)}")


# Main Execution
if __name__ == "__main__":
    # Enter the path to your CSV file
    csv_file_path = input("Enter the path to your CSV file: ").strip()

    try:
        # Create an instance of SupplyChainAlert and process the file
        alert_system = SupplyChainAlert()
        alert_system.process_csv_and_send_alerts(csv_file_path)
    except Exception as e:
        print(f"Error: {str(e)}")
