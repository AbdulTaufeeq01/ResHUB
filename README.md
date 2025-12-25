# Research Assistant 🤖

## What This Project Does

This is an AI-powered research assistant that answers questions about your documents. Upload your files (PDFs, text, or markdown), ask questions, and get answers with source citations. It works completely offline using local AI models.

**Think of it as a smart search engine for your documents.**

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

---

## Configuration

Edit `config.py` to customize settings:

```python
OLLAMA_MODEL_NAME = "llama3"      # Change AI model
RETRIEVER_TOP_K = 5                # How many sources to search
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
