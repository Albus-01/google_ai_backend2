from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
from fastapi.middleware.cors import CORSMiddleware

# PDF processing and file upload modules are no longer needed
from vector_database_module import VectorDB
from chatbot_logic import Chatbot

# --- Initialize the FastAPI app ---
app = FastAPI(
    title="AI Tutor Q&A API",
    description="An API for answering questions based on pre-ingested documents.",
    version="1.3.0"
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

# --- STARTUP LOGIC ---
@app.on_event("startup")
async def startup_event():
    """Initializes the database and chatbot clients on startup."""
    global chatbot_instance, vector_db_client
    print("--- 🚀 Starting API and Initializing AI Backend ---")
    try:
        vector_db_client = VectorDB()
        chatbot_instance = Chatbot(vector_db_client)
        print("\n--- ✅ AI Tutor is Ready ---")
    except Exception as e:
        print(f"❌ An error occurred during startup: {e}")
        
# --- API MODELS ---
class QuestionRequest(BaseModel):
    question: str

class AnswerResponse(BaseModel):
    answer: str

# --- API ENDPOINTS ---

# --- REMOVED: The /ingest endpoint has been deleted ---

@app.post("/ask", response_model=AnswerResponse)
async def ask_question(request: QuestionRequest):
    """Receives a question, uses the RAG chatbot to generate an answer."""
    if chatbot_instance is None:
        raise HTTPException(status_code=503, detail="Chatbot is not ready.")
    
    print(f"\nReceived question: {request.question}")
    final_answer = chatbot_instance.generate_answer(request.question)
    return AnswerResponse(answer=final_answer)

# --- MAIN EXECUTION (Unchanged) ---
if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)

