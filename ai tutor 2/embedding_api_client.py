import requests
import time
from configuration import GOOGLE_API_KEY

# The official model name for Google's text embedding model
MODEL_NAME = "gemini-embedding-001"
API_URL = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-embedding-001:embedContent"

def get_embedding_from_api(text: str, retries: int = 3):
    """
    Gets a text embedding from the Google AI (Gemini) API with retry logic.
    """
    if not GOOGLE_API_KEY:
        raise ValueError("Google AI API key is not configured. Please set GOOGLE_API_KEY in your .env file.")

    headers = {"Content-Type": "application/json"}
    payload = {
        'model': f'models/gemini-embedding-001',
        'content': {
            'parts': [{'text': text}]
        }
    }
    
    for attempt in range(retries):
        try:
            response = requests.post(f"{API_URL}?key={GOOGLE_API_KEY}", headers=headers, json=payload)
            response.raise_for_status()  # Raises HTTPError for bad responses (4xx or 5xx)
            
            # Extract the embedding from the response JSON
            embedding = response.json().get('embedding', {}).get('values')
            if embedding:
                return embedding
            else:
                print(f"    - API Warning: Response for chunk did not contain an embedding. Payload: {response.json()}")
                return None

        except requests.exceptions.RequestException as e:
            print(f"    API request failed on attempt {attempt + 1}: {e}")
            if attempt < retries - 1:
                print("    Retrying in 5 seconds...")
                time.sleep(5)  # Wait before retrying
            else:
                print(f"    Failed to get embeddings after {retries} attempts.")
                return None
    return None

