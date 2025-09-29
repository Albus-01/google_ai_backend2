import os
import google.generativeai as genai
from dotenv import load_dotenv

def list_available_models():
    """
    Connects to the Google AI API and lists all the models available
    for the configured API key, along with their supported methods.
    """
    print("--- Listing Available Google AI Models ---")
    
    # Load environment variables from a .env file.
    load_dotenv()
    print("[INFO] Loading API key from .env file...")

    try:
        api_key = os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            print("\n[ERROR] 'GOOGLE_API_KEY' not found. Please check your .env file.")
            return

        genai.configure(api_key=api_key)
        print("[INFO] API key configured. Fetching model list...")

        print("\n--- ✅ AVAILABLE MODELS ---")
        # List all models
        for model in genai.list_models():
            # We are interested in models that support the 'generateContent' method
            if 'generateContent' in model.supported_generation_methods:
                print(f"🔹 Model Name: {model.name}")
                print(f"   Description: {model.description}\n")
        
        print("----------------------------")

    except Exception as e:
        print("\n--- ❌ FAILURE! ---")
        print("An error occurred while trying to fetch the model list.")
        print("\n[ERROR DETAILS]:")
        print(e)

if __name__ == "__main__":
    list_available_models()