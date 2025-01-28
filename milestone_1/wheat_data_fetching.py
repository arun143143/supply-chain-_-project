import requests
import json
import os
import time
from typing import Dict, List, Optional
from datetime import datetime

class EventRegistryFetcher:
    """
    A class to fetch and store specific fields (title, body, date, sentiment) from Event Registry API.
    """
    def __init__(self, api_key: str):
        self.api_url = "https://eventregistry.org/api/v1/article/getArticles"
        self.api_key = api_key
        self.headers = {"Content-Type": "application/json"}

    def _process_article(self, article: Dict) -> Dict:
        """
        Extract only title, body, date, and sentiment from an article.
        """
        return {
            'title': article.get('title', ''),
            'body': article.get('body', ''),
            'date': article.get('date', ''),
            'sentiment': article.get('sentiment', 0)
        }

    def fetch_data(self, 
                  query_terms: List[str],
                  output_file: str,
                  max_pages: int = 130,
                  articles_per_page: int = 100,
                  language: str = "eng") -> None:
        """
        Fetch data from Event Registry API and save specific fields.
        
        Args:
            query_terms: List of keywords to search for
            output_file: Path to save the processed data
            max_pages: Maximum number of pages to fetch
            articles_per_page: Number of articles per page
            language: Language code for articles
        """
        query_payload = {
            "query": {
                "$query": {
                    "$and": [
                        {"$or": [{"keyword": term, "keywordSearchMode": "exact"} 
                                for term in query_terms]},
                        {"lang": language}
                    ]
                },
                "$filter": {
                    "forceMaxDataTimeWindow": "31",
                    "dataType": ["news", "blog"],
                    "isDuplicate": "skipDuplicates"
                }
            },
            "resultType": "articles",
            "articlesSortBy": "date",
            "includeArticleEventUri": False,
            "includeConceptLabel": False,
            "apiKey": self.api_key
        }

        processed_articles = []
        
        try:
            for page in range(1, max_pages + 1):
                query_payload["articlesPage"] = page
                query_payload["articlesCount"] = articles_per_page

                response = requests.post(
                    self.api_url,
                    headers=self.headers,
                    json=query_payload,
                    stream=True
                )
                response.raise_for_status()

                if 'application/json' not in response.headers.get('Content-Type', ''):
                    print(f"Unexpected content type: {response.headers.get('Content-Type')}")
                    break

                data = response.json()
                # Add this line to check the structure of the 'articles' field in the response
                articles = data.get('articles', {}).get('results', []) 
                # If articles is a string, try to parse it as JSON, otherwise treat it as a list of dictionaries
                if isinstance(articles, str):
                    try:
                        articles = json.loads(articles)
                    except json.JSONDecodeError:
                        print(f"Could not parse 'articles' as JSON on page {page}")
                        articles = [] # Set to empty if not parsable
                
                if not articles:
                    print("No more articles to fetch.")
                    break

                # Process articles to get only required fields
                processed_articles.extend([
                    self._process_article(article) for article in articles
                ])

                print(f"Processed page {page} ({len(articles)} articles)")
                time.sleep(1)  # Rate limiting

            # Save processed data
            output_data = {
                'metadata': {
                    'query_terms': query_terms,
                    'total_articles': len(processed_articles),
                    'fetch_date': datetime.now().isoformat()
                },
                'articles': processed_articles
            }

            # Save with both JSON and pickle formats
            self._save_data(output_data, output_file)
            
            print(f"Successfully processed and saved {len(processed_articles)} articles")

        except requests.exceptions.RequestException as e:
            print(f"Error occurred during fetch: {e}")
            
    def _save_data(self, data: Dict, base_output_file: str) -> None:
        """Save data in both JSON and pickle formats."""
        # Save as JSON for human readability
        json_path = f"{base_output_file}.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        # Save as pickle for efficient machine reading
        pickle_path = f"{base_output_file}.pkl"
        import pickle
        with open(pickle_path, 'wb') as f:
            pickle.dump(data, f)

    @staticmethod
    def load_data(file_path: str) -> Optional[Dict]:
        """
        Load processed data from either JSON or pickle format.
        """
        if file_path.endswith('.json'):
            with open(file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        elif file_path.endswith('.pkl'):
            import pickle
            with open(file_path, 'rb') as f:
                return pickle.load(f)
        else:
            raise ValueError("File must be either .json or .pkl format")

# Example usage
if __name__ == "__main__":
    API_KEY = "3c92e915-5ee3-44b8-82c8-4017ce33328f"  # Replace with your API key
    fetcher = EventRegistryFetcher(API_KEY)
    
    # Fetch data
    output_file = os.path.join(os.getcwd(), "wheat_data")
    fetcher.fetch_data(
        query_terms=["Wheat"],
        output_file=output_file,
        max_pages=130,
        articles_per_page=100
    )
    
    # Example of how to access the simplified data
    data = EventRegistryFetcher.load_data(f"{output_file}.json")
    for article in data['articles'][:3]:  # Print first 3 articles as example
        print(f"\nTitle: {article['title']}")
        print(f"Date: {article['date']}")
        print(f"Sentiment: {article['sentiment']}")
        print(f"Body preview: {article['body'][:100]}...")  # First 100 chars of body