from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn

from vector_database_module import VectorDB
from chatbot_logic import Chatbot

app = FastAPI(
    title="AI Tutor Q&A API",
    description="An API for answering questions based on pre-ingested documents.",
    version="1.3.0"
)

# --- MODIFIED: Initialize clients as None ---
chatbot_instance: Chatbot = None
vector_db_client: VectorDB = None

# --- This function ensures clients are loaded before a request ---
def get_chatbot_instance():
    """Initializes and returns the chatbot instance, creating it if it doesn't exist."""
    global chatbot_instance, vector_db_client
    if chatbot_instance is None:
        print("--- LAZY LOADING: Initializing AI Backend for first request ---")
        try:
            vector_db_client = VectorDB()
            chatbot_instance = Chatbot(vector_db_client)
            print("--- ✅ AI Backend is now Ready ---")
        except Exception as e:
            print(f"❌ An error occurred during lazy initialization: {e}")
            raise HTTPException(status_code=503, detail="Could not initialize AI backend.")
    return chatbot_instance

# --- REMOVED: @app.on_event("startup") is no longer needed ---

# --- API MODELS (Unchanged) ---
class QuestionRequest(BaseModel):
    question: str

class AnswerResponse(BaseModel):
    answer: str

# --- API ENDPOINTS ---
@app.post("/ask", response_model=AnswerResponse)
async def ask_question(request: QuestionRequest):
    """Receives a question, uses the RAG chatbot to generate an answer."""
    # MODIFIED: Get the instance, which will trigger initialization on first call
    chatbot = get_chatbot_instance()
    
    if chatbot is None:
        raise HTTPException(status_code=503, detail="Chatbot is not ready.")
    
    print(f"\nReceived question: {request.question}")
    final_answer = chatbot.generate_answer(request.question)
    return AnswerResponse(answer=final_answer)

# --- MAIN EXECUTION (Unchanged) ---
if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
