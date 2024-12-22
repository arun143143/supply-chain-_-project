import pandas as pd
from textblob import TextBlob
import concurrent.futures
import time

# Step 1: Load the JSON file into a DataFrame
json_file_path = r'C:\Users\arunp\OneDrive\Desktop\infosys internship\project\combined_data.json'  # Provide the correct file path for your JSON file
df = pd.read_json(json_file_path)

# Assuming 'df' is your original DataFrame containing the JSON data
df_normalized = pd.json_normalize(df['event_registry'][0]['articles']['results'])

# Filter the columns you need ('title', 'body', 'date', 'time')
df_filtered = df_normalized[['title', 'body', 'date', 'time']]

# Display the filtered DataFrame
print("Filtered DataFrame:")
print(df_filtered.head())

# Step 2: Perform sentiment analysis on the 'title' column
def get_sentiment_score(text):
    if isinstance(text, str):  # Ensure the text is a string
        analysis = TextBlob(text)
        return analysis.sentiment.polarity
    return 0  # Return 0 if text is not valid for sentiment analysis

# Step 3: Apply sentiment analysis in parallel for 'title' and 'body' columns
def apply_sentiment_in_parallel(df, column_name):
    with concurrent.futures.ThreadPoolExecutor() as executor:
        results = list(executor.map(get_sentiment_score, df[column_name]))
    return results

# Apply sentiment analysis in parallel
start_time = time.time()
df_filtered['title_sentiment'] = apply_sentiment_in_parallel(df_filtered, 'title')
df_filtered['body_sentiment'] = apply_sentiment_in_parallel(df_filtered, 'body')

# Calculate overall sentiment if both title and body sentiment are available
if 'title_sentiment' in df_filtered.columns and 'body_sentiment' in df_filtered.columns:
    df_filtered['overall_sentiment'] = df_filtered[['title_sentiment', 'body_sentiment']].mean(axis=1)

# Step 4: Select only the final columns you want to store
columns_to_store = ['overall_sentiment', 'title', 'body', 'date', 'time']
df_final = df_filtered[columns_to_store]

# Step 5: Save the DataFrame to a CSV file with gzip compression
output_file_path = 'processed_data_with_sentiment.csv.gz'
df_final.to_csv(output_file_path, index=False, compression='gzip')

# Print success message and file path
print(f"Processed data with sentiment categories saved to {output_file_path}.")
print(f"Execution Time: {time.time() - start_time} seconds")
