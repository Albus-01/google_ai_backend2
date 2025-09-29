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
        
        if not context_chunks:
            return "I'm sorry, I couldn't find any relevant information in the document to answer that question."

        # Combine the retrieved text chunks into a single block of context
        combined_context = "\n\n".join([chunk['text'] for chunk in context_chunks])
        
        # Create a detailed prompt for the generative model
        prompt = f"""
        Based *only* on the following context from a document, please provide a clear and concise answer to the user's question.
        Output the context if the score is high as 0.9 and then you get and then two lines below use that context and use external knowledge to output a precise answer.Output a solution always(only context info is optional)
        
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

