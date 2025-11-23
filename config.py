from pathlib import Path

# Base project directory
BASE_DIR=Path(__file__).resolve().parent

# Directory for storing the pdf files
DATA_DIR=BASE_DIR/"data"

# Directory for storing vector data
CHROMA_DB_DIR=BASE_DIR/"chroma_db"

# Embedding model
EMBEDDING_MODEL_NAME="sentence-transformers/all-MiniLM-L6-v2"

# LLM Model used by ollama
OLLAMA_MODEL_NAME="llama3"

# Numbder of chunks to retrieve from vector database
RETRIEVER_TOP_K=5

"""
KEEPS ALL THE CONFIGURATION SETTINGS IN PLACE
IF THE MODEL NAMES NEEDED TO BE CHANGED IT CAN BE DONE HERE
"""