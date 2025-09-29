import os
from dotenv import load_dotenv

# Load environment variables from a .env file for local development
load_dotenv()

# --- Pinecone Configuration ---
PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
PINECONE_INDEX_NAME = os.environ.get('PINECONE_INDEX_NAME')

# --- Google AI Configuration ---
# Your new API key for the Gemini API, obtained from https://aistudio.google.com/
GOOGLE_API_KEY = os.environ.get('GOOGLE_API_KEY')

# --- Local File Configuration ---
# IMPORTANT: Place the PDF you want to process in the same directory as main.py
# and update this variable with its name.