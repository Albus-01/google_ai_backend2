from pinecone import Pinecone
from configuration import PINECONE_API_KEY, PINECONE_INDEX_NAME
from embedding_api_client import get_embedding_from_api

class VectorDB:
    """A wrapper class for Pinecone operations using the new SDK."""
    
    def __init__(self):
        if not all([PINECONE_API_KEY, PINECONE_INDEX_NAME]):
            raise ValueError("Pinecone environment variables are not fully configured.")
        
        # Initialize the Pinecone client with your API key
        pc = Pinecone(api_key=PINECONE_API_KEY)
        
        # Connect to your specific index
        self.index = pc.Index(PINECONE_INDEX_NAME)
        print("✅ Pinecone database initialized successfully.")

    # --- REMOVED: The upsert method is no longer part of this class ---

    def query(self, question: str, top_k: int = 3):
        """
        Generates an embedding for a question using the Google AI API and queries the index.
        """
        print(f"  - Generating embedding for the question...")
        question_embedding = get_embedding_from_api(question)
        
        # CRITICAL FIX: Ensure the embedding is valid before querying.
        if not question_embedding or not isinstance(question_embedding, list):
            print("    - Could not generate a valid embedding for the question. The API might be temporarily unavailable.")
            return []
            
        print("  - Querying vector database for relevant context...")
        query_result = self.index.query(
            vector=question_embedding,
            top_k=top_k,
            include_metadata=True
        )
        
        # Extract and return the text from the query results
        matches = query_result.get('matches', [])
        return [
            {'text': match['metadata']['text'], 'source': match['metadata'].get('source', 'N/A')}
            for match in matches
        ]