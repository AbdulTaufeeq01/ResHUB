# Research Assistant

A comprehensive research assistant application built with LangChain, LangGraph, and Streamlit. This system enables intelligent document processing, question answering, and multi-agent research capabilities.

## 🚀 Quick Start (2 Minutes)

### 1. Activate Virtual Environment
```bash
researchvenv\Scripts\activate
```

### 2. Index Your Documents
```bash
python ingestion_build_index.py
```

### 3. Launch the App
```bash
streamlit run app_streamlit.py
```

Open your browser to `http://localhost:8501` and start using the assistant!

---

## Features

### 1. **RAG Question Answering**
- Ask questions about your documents
- Get answers with source citations
- Semantic similarity search
- Conversation history tracking

**Sample Questions:**
```
"What is the vacation policy?"
"How do I create a list in Python?"
"Which cloud platform is best for ML?"
```

### 2. **Multi-Agent Research**
- 3 specialized agents working together:
  - **Query Analyzer**: Refines questions for better retrieval
  - **Researcher**: Searches vector database
  - **Synthesizer**: Combines findings into comprehensive answers
- Perfect for complex, open-ended research questions

**Sample Questions:**
```
"How can we improve employee engagement?"
"What are best practices for project success?"
"Compare cloud platforms for enterprise use"
```

### 3. **Document Management**
- Support for PDF, TXT, and Markdown files
- Automatic chunking and embedding generation
- View indexed documents

---

## Installation

### Prerequisites
- Python 3.9+
- Ollama (for local LLM)
- Virtual environment recommended

### Setup

1. **Create Virtual Environment**
```bash
python -m venv researchvenv
researchvenv\Scripts\activate  # Windows
# or
source researchvenv/bin/activate  # macOS/Linux
```

2. **Install Dependencies**
```bash
pip install -r requirements.txt
```

3. **Download Ollama Model** (if needed)
```bash
ollama pull llama3
```

---

## Project Structure

```
research_assistant/
├── app_streamlit.py              # Web interface
├── ingestion_build_index.py      # Document ingestion
├── rag_qa.py                     # QA system
├── graph_multi_agent.py          # Multi-agent research
├── view_chroma_db.py             # Vector DB viewer
├── config.py                     # Configuration
├── requirements.txt              # Dependencies
├── README.md                     # This file
├── data/                         # Your documents
├── chroma_db/                    # Vector database
└── researchvenv/                 # Virtual environment
```

---

## Using the Web Interface

### Tab 1: RAG QA 📚
- Ask specific questions about documents
- Immediate answers with sources
- Best for factual lookups

### Tab 2: Upload Documents 📤
- Add new PDF, TXT, or MD files
- Auto-indexes for searching

### Tab 3: Multi-Agent Research 🤖
- Complex research queries
- Watch 3 agents collaborate
- Comprehensive synthesized answers

---

## Configuration

Edit `config.py` to customize:

```python
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
OLLAMA_MODEL_NAME = "llama3"
RETRIEVER_TOP_K = 5
```

---

## Sample Data Included

Your `data/` folder contains 7 documents for testing:

| Document | Type | Topic |
|----------|------|-------|
| company_handbook.txt | Corporate | HR policies & benefits |
| python_guide.txt | Technical | Python fundamentals |
| cloud_computing.txt | Comparison | AWS vs Azure vs GCP |
| project_management_faq.txt | FAQ | PM best practices |
| ai_ml_guide.txt | Educational | AI & machine learning |
| meeting_notes.txt | Business | Strategic planning |
| research_paper_2.pdf | Academic | Research paper |

---

## Command Line Usage

```bash
# Build index from documents
python ingestion_build_index.py

# Run RAG QA (interactive)
python rag_qa.py

# Run multi-agent research
python graph_multi_agent.py

# View vector database
python view_chroma_db.py
```

---

## Example Queries

### Company/HR Questions
- "What is the vacation policy?"
- "How much health insurance does the company provide?"
- "What is the remote work policy?"

### Programming Questions
- "How do I create a list in Python?"
- "What's the difference between lists and dictionaries?"
- "How do I handle errors?"

### Cloud Computing
- "What are the main cloud providers?"
- "How do AWS and Azure compare?"
- "Which cloud is best for AI/ML?"

### Project Management
- "What is scope creep?"
- "How should I manage project risks?"
- "What makes a project successful?"

### AI & Machine Learning
- "What is machine learning?"
- "Explain supervised vs unsupervised learning"
- "How do neural networks work?"

### Multi-Agent Research
- "How can we improve employee engagement and development?"
- "What are the key strategic priorities for 2025?"
- "Compare different approaches to cloud adoption"

---

## API Usage

### In Python Code

```python
# Build index
from ingestion_build_index import build_index
build_index()

# Use RAG QA
from rag_qa import get_qa_chain
chain = get_qa_chain()
result = chain.invoke("Your question here")
print(result)

# Use Multi-Agent Research
from graph_multi_agent import run_research_assistant
result = run_research_assistant("Your research query")
print(result["final_answer"])
```

---

## Models & Technologies

- **LLM**: Ollama Llama3 (local, private)
- **Embeddings**: HuggingFace all-MiniLM-L6-v2
- **Vector DB**: Chroma (persistent)
- **Framework**: LangChain + LangGraph
- **UI**: Streamlit

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| Ollama not found | Install Ollama, run `ollama serve` |
| No documents found | Add files to `data/` folder |
| Port 8501 in use | `streamlit run app_streamlit.py --server.port=8502` |
| Slow response | Ollama needs more resources or is indexing |

---

## Performance

- **Indexing**: ~100 PDFs in 2-5 minutes
- **Search**: <1 second
- **Response**: 5-30 seconds (LLM generation)
- **Memory**: ~2GB (embeddings + Ollama)

---

## Adding Your Own Documents

1. Place PDF, TXT, or MD files in the `data/` folder
2. Run: `python ingestion_build_index.py`
3. Use the web interface to ask questions

---

## System Requirements

- **RAM**: 4GB minimum (8GB+ recommended)
- **Disk**: 5GB for Ollama + embeddings
- **Internet**: For downloading models on first run

---

## Support

For issues or questions, check:
- Configuration in `config.py`
- Individual module documentation
- Error messages in terminal output
