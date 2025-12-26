# Research Assistant

## What This Project Does

This is an AI-powered research assistant that answers questions about your documents. Upload your files (PDFs, text, or markdown), ask questions, and get answers with source citations. It works completely offline using local AI models.

**Think of it as a smart search engine for your documents.**

### How It Works

1. **Document Ingestion** - You add documents (PDFs, text files, etc.) to the system
2. **Smart Indexing** - The AI breaks documents into chunks and creates embeddings (digital fingerprints) to understand content
3. **Vector Database** - These embeddings are stored in a searchable database (Chroma)
4. **Question Answering** - When you ask a question, the system:
   - Searches for relevant document chunks
   - Sends them to a local AI model
   - Returns an answer with sources cited

### Key Features

- **Completely Offline** - Runs on your computer, no data sent to the cloud
- **Multi-Agent Research** - 3 AI agents work together for complex questions
- **Web Interface** - Easy-to-use Streamlit app (no coding needed)
- **Source Citations** - Know exactly where answers come from
- **Conversation History** - Track all your questions and answers

---

## Quick Start (3 Steps)

### 1. Start the Virtual Environment
```bash
researchvenv\Scripts\activate
```

### 2. Index Your Documents
```bash
python ingestion_build_index.py
```

### 3. Open the Web App
```bash
streamlit run app_streamlit.py
```
Then visit `http://localhost:8501` in your browser.

---

## What Can You Do?

**Ask Questions About Your Documents**
- "What is the vacation policy?"
- "How do I create a list in Python?"
- "Which cloud platform is best for ML?"

**Complex Research Questions** (uses multiple AI agents working together)
- "How can we improve employee engagement?"
- "Compare cloud platforms for enterprise use"
- "What are best practices for project success?"

**Upload New Documents**
- Add PDF, TXT, or Markdown files through the web interface
- They're automatically processed and searchable

---

## How to Set Up (First Time Only)

1. **Create a Python virtual environment** (if you haven't already)
```bash
python -m venv researchvenv
researchvenv\Scripts\activate
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Install Ollama** (the AI engine)
   - Download from https://ollama.ai
   - Run: `ollama pull llama3`

---

## Technology Stack

### AI & Machine Learning
- **Ollama + Llama3** - Local AI language model (no cloud required)
- **HuggingFace Embeddings** - Converts text to vector embeddings
- **Chroma** - Vector database for semantic search
- **LangChain** - Framework for building LLM applications
- **LangGraph** - Multi-agent orchestration system

### Frontend & Backend
- **Streamlit** - Web interface (Python-based, no JavaScript needed)
- **Python 3.9+** - Programming language

### Why These Technologies?

| Technology | Why We Use It |
|------------|---------------|
| **Ollama** | Runs AI locally (private, no internet needed) |
| **Chroma** | Super fast semantic search on document chunks |
| **LangChain** | Handles all the LLM complexity for us |
| **LangGraph** | Lets multiple AI agents work together |
| **Streamlit** | Fast web app development without web dev knowledge |

---

## Project Structure

```
research_assistant/
├── app_streamlit.py          # Web interface (what you see in browser)
├── ingestion_build_index.py  # Processes your documents
├── rag_qa.py                 # Q&A engine
├── graph_multi_agent.py      # Multi-agent research system
├── config.py                 # Settings you can change
├── data/                     # Your documents go here
├── chroma_db/                # Where answers are stored
└── requirements.txt          # Python packages needed
```

### What Each File Does

- **app_streamlit.py** - The web app you interact with
- **ingestion_build_index.py** - Reads your documents and creates the searchable index
- **rag_qa.py** - Simple Q&A system for direct questions
- **graph_multi_agent.py** - Advanced system using 3 AI agents for complex research
- **config.py** - Settings like which AI model to use, how many search results to return

---

## Configuration

Edit `config.py` to customize settings:

```python
OLLAMA_MODEL_NAME = "llama3"      # Change AI model (llama3, mistral, etc)
RETRIEVER_TOP_K = 5                # How many document chunks to search
EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # Embedding model to use
```

---

## Command Line Tools

**Add documents to the index:**
```bash
python ingestion_build_index.py
```

**Ask questions from command line:**
```bash
python rag_qa.py
```

**Complex research queries:**
```bash
python graph_multi_agent.py
```

---

## Sample Documents Included

The `data/` folder has 7 test documents:
- **company_handbook.txt** - HR policies
- **python_guide.txt** - Python basics
- **cloud_computing.txt** - AWS vs Azure vs GCP
- **project_management_faq.txt** - PM tips
- **ai_ml_guide.txt** - AI & ML explained
- **meeting_notes.txt** - Strategic notes
- **research_paper_2.pdf** - Sample research paper

Try asking about these to test the system!

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| Ollama not found | Install from https://ollama.ai and run `ollama serve` |
| No documents found | Add files to `data/` folder, then run ingestion script |
| Port 8501 in use | Run: `streamlit run app_streamlit.py --server.port=8502` |
| Slow responses | Wait - the AI is generating answers (can take 10-30 seconds) |

---

## What You Need

- **Python 3.9+**
- **4GB+ RAM** (8GB recommended)
- **5GB disk space**
- **Ollama** (free AI engine)
