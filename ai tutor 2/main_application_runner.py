# main_application_runner.py

import os
from configuration import PDF_FILE_PATH
# pdf_processing_module is no longer needed here
from vector_database_module import VectorDB
from chatbot_logic import Chatbot

def main():
    """
    MODIFIED: Orchestrates the chatbot interaction, assuming data is already in the DB.
    """
    print("--- Starting AI Backend Pipeline ---")

    # 1. Initialize the vector database client
    print("Initializing vector database...")
    try:
        vector_db_client = VectorDB()
    except Exception as e:
        print(f"Error initializing vector database: {e}")
        return
    print("✅ Vector database initialized.")

    # --- REMOVED THE DOCUMENT PROCESSING AND UPSERTING LOGIC ---
    # The app now assumes the data is already in Pinecone.

    # 2. Initialize the full RAG chatbot
    print("\n--- Chatbot is Ready ---")
    print("You can now ask questions about the document. Type 'exit' to quit.")
    chatbot = Chatbot(vector_db_client)

    # 3. Start the interactive chat loop
    while True:
        question = input("\nYour Question: ")
        if question.lower() == 'exit':
            break
        
        print("Thinking...")
        final_answer = chatbot.generate_answer(question)
        
        print("\n--- Answer ---")
        print(final_answer)

    print("\n--- Chatbot session ended ---")


if __name__ == "__main__":
    main()
