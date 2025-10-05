import google.generativeai as genai
from configuration import GOOGLE_API_KEY
from vector_database_module import VectorDB

# Configure the generative AI model using the key from your .env file
if not GOOGLE_API_KEY:
    raise ValueError("Google AI API key is not configured. Please set GOOGLE_API_KEY in your .env file.")
genai.configure(api_key=GOOGLE_API_KEY)

class Chatbot:
    """
    Handles the full RAG pipeline: retrieval of context and generation of an answer.
    """
    
    def __init__(self, vector_db_client: VectorDB):
        self.vector_db_client = vector_db_client
        # Initialize the generative model. 'gemini-1.5-flash' is fast and capable.
        self.generative_model = genai.GenerativeModel('models/gemini-2.5-flash')

    def _get_relevant_context(self, question: str):
        """
        Internal method to retrieve relevant context from the vector database.
        This is the "Retrieval" part of RAG.
        """
        return self.vector_db_client.query(question)

    def generate_answer(self, question: str):
        """
        Performs the full RAG process to generate a conversational answer.
        This is the "Generation" part of RAG.
        """
        print("  - Retrieving relevant context from the document...")
        context_chunks = self._get_relevant_context(question)
        

        # Combine the retrieved text chunks into a single block of context
        combined_context = "\n\n".join([chunk['text'] for chunk in context_chunks])
        
        # Create a detailed prompt for the generative model
        prompt = f"""
        Use the provided context if it is useful to give precise accurate answer to the prompt.
        
        --- CONTEXT ---
        {combined_context}
        --- END OF CONTEXT ---
        
        USER'S QUESTION:
        {question}
        
        ANSWER:
        """
        
        print("  - Sending context and question to the generative model...")
        try:
            # Generate the final answer using the Gemini model
            response = self.generative_model.generate_content(prompt)
            return response.text
        except Exception as e:
            print(f"    Error during answer generation: {e}")
            return "I encountered an error while trying to formulate an answer. The API may be unavailable or the content might be blocked."



