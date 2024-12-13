"""from eventregistry import EventRegistry, QueryArticlesIter, QueryItems

# Initialize the Event Registry API client with your API key
er = EventRegistry(apiKey = '62ccd8cc-37bf-49d2-b6d7-915c3712459d')

# Define the keywords related to Supply Chain Management (adjust as needed)
keywords = QueryItems.OR([
    "supply chain"
])

# Define the relevant categories (adjust as needed)
categoryUris = [
    er.getCategoryUri("news/business"),  # Business-related articles
    er.getCategoryUri("news/economy"),   # Economy-related articles
    er.getCategoryUri("news/logistics"), # Logistics-related articles
    er.getCategoryUri("news/technology"), # Technology-related articles
    er.getCategoryUri("news/war"),
    er.getCategoryUri("news/geo politics"),
    er.getCategoryUri("news/warehouse"),
    er.getCategoryUri("news/inventory"),
    er.getCategoryUri("news/technology"),
    er.getCategoryUri("news/supply chain resilience"),
    er.getCategoryUri("news/supplier management"),
    er.getCategoryUri("news/demand forecasting"),
    
]

# Filter articles in English language and from the last year
q = QueryArticlesIter(
    keywords = keywords,                  # Supply chain related keywords
    lang = "eng",                          # Filter for English language
    dateStart = "2023-12-01",              # Start date (1 year ago from now)
    dateEnd = "2024-12-01",                # End date (current date)
    categoryUri = QueryItems.OR(categoryUris)  # Add the selected categories
)


import sys
import io

# Reconfigure stdout to UTF-8
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Execute the query and print the results
for art in q.execQuery(er, sortBy = "date", maxItems = 1000):  # Get up to 500 articles, adjust as needed
    print(art)


# save the file in local mechin
import json
import os

# ... (Your existing code to fetch and populate 'results') ...

# Specify the file path
file_path = 'eventregistry_results.json'  

# Save the JSON data to the file
with open(file_path, 'w') as f:
    json.dump(art, f, indent=4)

# Print a message indicating where the file was saved
print(f"Results saved to: {os.path.abspath(file_path)}")"""



import requests
import json
import os

def fetch_data(api_name, api_url, query_params):
    """Fetch data from an API and yield results to minimize memory usage."""
    try:
        with requests.get(api_url, params=query_params, stream=True) as response:
            response.raise_for_status()
            yield {api_name: response.json()}  # Stream data to avoid large memory allocation
    except requests.exceptions.RequestException as e:
        print(f"Error fetching from {api_name}: {e}")

# API configurations
apis = [
    {
        "name": "event_registry",
        "url": "https://eventregistry.org/api/v1/article/getArticles",
        "key": "3c92e915-5ee3-44b8-82c8-4017ce33328f",
        "params": {
            "q": "supply chain OR logistics OR geopolitics",
            "from": "2023-01-01",
            "to": "2024-12-12",
            "sortBy": "relevancy",
            "lang": ["eng"],
            "pageSize": 100,
        }
    },
    {
        "name": "news_api",
        "url": "https://newsapi.org/v2/everything",
        "key": "6a5c75a5361d431580636fc6ee43d315",
        "params": {
            "q": "supply chain AND FMCG AND shipping",
            "sortBy": "relevancy",
            "language": "en",
            "pageSize": 100,
        }
    }
]

# Fetch and save data in chunks to minimize memory usage
output_file = os.path.join(os.getcwd(), "combined_data.json")

with open(output_file, "w", encoding="utf-8") as file:
    file.write("[")  # Start JSON array
    first_entry = True
    for api in apis:
        api["params"]["apiKey"] = api["key"]
        for result in fetch_data(api["name"], api["url"], api["params"]):
            if not first_entry:
                file.write(",\n")  # Add comma between entries
            json.dump(result, file, indent=4)
            first_entry = False
    file.write("]")  # Close JSON array

print(f"Data saved to {output_file}")
