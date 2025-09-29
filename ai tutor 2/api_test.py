import requests

# --- CONFIGURATION ---

# 1. Paste your NEW Hugging Face token (with 'write' permissions) here.
HF_API_TOKEN = "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" 

# 2. This is the stable model URL we are testing against.
API_URL = "https://api-inference.huggingface.co/models/sentence-transformers/all-MiniLM-L6-v2"

# 3. This is a simple test sentence.
TEST_SENTENCE = "This is a test sentence."

# --- TEST SCRIPT ---

print("--- Starting Hugging Face API Connection Test ---")

if not HF_API_TOKEN or "YOUR_NEW" in HF_API_TOKEN:
    print("\nERROR: Please paste your new Hugging Face API token into the HF_API_TOKEN variable in this script.")
else:
    headers = {"Authorization": f"Bearer {HF_API_TOKEN}"}
    payload = {"inputs": [TEST_SENTENCE]}

    print(f"Sending request to: {API_URL}")
    print(f"Using token starting with: {HF_API_TOKEN[:4]}...")

    try:
        response = requests.post(API_URL, headers=headers, json=payload)
        response.raise_for_status()  # This will raise an error for 4xx or 5xx responses

        print("\n✅ SUCCESS! Connection to Hugging Face API is working correctly.")
        print("The problem is likely a configuration issue in your main project (e.g., the .env file).")
        print("\nReceived Response (first 100 characters):")
        print(str(response.json())[:100] + "...")

    except requests.exceptions.RequestException as e:
        print(f"\n❌ FAILURE! The API request failed.")
        print(f"Error details: {e}")
        print("\nThis confirms the issue is with your API token or your network connection.")
        print("Please double-check the token you just generated and ensure it has 'write' permissions.")

print("\n--- Test Complete ---")