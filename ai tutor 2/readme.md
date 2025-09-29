# Modular AI Backend for PDF Processing and Q&A

This project is a pure Python backend designed to process PDF documents, store their content in a vector database (Pinecone), and retrieve relevant information to answer user questions. It has been refactored from a web application into a modular, command-line-driven system suitable for backend AI tasks.

## Architecture

The project is broken down into several distinct modules, promoting separation of concerns and maintainability:

-   `main.py`: The orchestrator and main entry point for running the pipeline.
-   `config.py`: Centralizes all configuration, loading API keys and settings from a `.env` file.
-   `pdf_processor.py`: Contains all logic for reading, extracting text from, and chunking PDF files.
-   `embedding_client.py`: A client dedicated to interacting with the Hugging Face Inference API to generate text embeddings.
-   `vector_db.py`: A class-based module to handle all communication with Pinecone, including upserting data and querying.
-   `chatbot.py`: Implements the retrieval logic. It uses the `vector_db` module to find context relevant to a user's question.

## Setup and Usage

### 1. Prerequisites

-   Python 3.8+
-   A [Pinecone](https://app.pinecone.io/) account with an index created. You will need your **API Key**, **Environment**, and **Index Name**.
-   A [Hugging Face](https://huggingface.co/settings/tokens) account. You will need an **Access Token** with `read` permissions.

### 2. Installation

Clone the repository and install the required dependencies:

```bash
git clone <your-repo-url>
cd <your-repo-name>
pip install -r requirements.txt