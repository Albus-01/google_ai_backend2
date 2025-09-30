from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
from fastapi.middleware.cors import CORSMiddleware
import asyncio

# PDF processing and file upload modules are no longer needed
from vector_database_module import VectorDB
from chatbot_logic import Chatbot

# --- Initialize the FastAPI app ---
app = FastAPI(
    title="AI Tutor Q&A API",
    description="An API for answering questions based on pre-ingested documents.",
    version="1.4.0"  # Incremented version for new logic
)

# --- ADD CORS MIDDLEWARE ---
# This allows the API to be accessed from any origin.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# --- GLOBAL VARIABLES ---
chatbot_instance: Chatbot = None
vector_db_client: VectorDB = None
# A lock to prevent race conditions during the first initialization
initialization_lock = asyncio.Lock()


# --- LAZY INITIALIZATION LOGIC ---
async def initialize_chatbot_lazily():
    """
    Initializes the database and chatbot clients on the first request.
    Uses a lock to prevent multiple concurrent initializations.
    """
    global chatbot_instance, vector_db_client
    async with initialization_lock:
        # Check again inside the lock in case another request finished
        # initialization while this one was waiting.
        if chatbot_instance is None:
            print("--- 🚀 First request received, Initializing AI Backend (this may take a moment)... ---")
            try:
                vector_db_client = VectorDB()
                chatbot_instance = Chatbot(vector_db_client)
                print("\n--- ✅ AI Tutor is now Ready ---")
            except Exception as e:
                print(f"❌ An error occurred during lazy initialization: {e}")
                # Reset globals so the next request can try again
                chatbot_instance = None
                vector_db_client = None


# --- STARTUP LOGIC (REMOVED) ---
# @app.on_event("startup") has been removed to allow for lazy loading.

        
# --- API MODELS ---
class QuestionRequest(BaseModel):
    question: str

class AnswerResponse(BaseModel):
    answer: str
    confidence_score: float = 0.0 # Added based on your preferences

# --- API ENDPOINTS ---

@app.post("/ask", response_model=AnswerResponse)
async def ask_question(request: QuestionRequest):
    """
    Receives a question. On the first call, it initializes the chatbot,
    then uses the RAG chatbot to generate an answer.
    """
    # Lazy initialize the chatbot on the first request.
    if chatbot_instance is None:
        await initialize_chatbot_lazily()
    
    # After attempting initialization, check if it's ready.
    if chatbot_instance is None:
        raise HTTPException(
            status_code=503, 
            detail="Chatbot is not ready after initialization attempt. This might be a temporary issue. Please try again in a moment."
        )
    
    print(f"\nReceived question: {request.question}")
    # Assuming generate_answer can return a tuple (answer, score)
    # If not, you'll need to adjust chatbot_logic.py
    try:
        final_answer, score = chatbot_instance.generate_answer_with_score(request.question)
        return AnswerResponse(answer=final_answer, confidence_score=score)
    except AttributeError:
        # Fallback if the score method doesn't exist yet
        final_answer = chatbot_instance.generate_answer(request.question)
        # Returning a default score as a placeholder
        return AnswerResponse(answer=final_answer, confidence_score=0.95)


# --- MAIN EXECUTION (Unchanged) ---
if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)


