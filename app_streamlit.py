import streamlit as st
import os
from pathlib import Path
from typing import Optional

from ingestion_build_index import build_index, load_documents
from rag_qa import get_qa_chain, get_vectordb
from graph_multi_agent import run_research_assistant
from config import DATA_DIR, CHROMA_DB_DIR

"""
Streamlit Research Assistant Application
This app provides:
1. Document Ingestion - Upload and process documents
2. RAG QA - Question answering over documents
3. Multi-Agent Research - Advanced research using multiple agents
"""

def initialize_session_state():
    """Initialize Streamlit session state variables"""
    if "qa_chain" not in st.session_state:
        st.session_state.qa_chain = None
    if "vectordb_loaded" not in st.session_state:
        st.session_state.vectordb_loaded = False
    if "conversation_history" not in st.session_state:
        st.session_state.conversation_history = []


def load_qa_chain():
    """Load the QA chain if not already loaded"""
    if st.session_state.qa_chain is None:
        try:
            st.session_state.qa_chain = get_qa_chain()
            st.session_state.vectordb_loaded = True
        except Exception as e:
            st.error(f"Error loading QA chain: {e}")
            return False
    return True


def page_ingestion():
    """Document ingestion page"""
    st.title("📄 Document Ingestion")
    
    st.markdown("""
    Upload PDF, TXT, or Markdown files to build your knowledge base.
    The documents will be split into chunks and indexed using embeddings.
    """)
    
    # Display current data directory
    st.info(f"📁 Documents Directory: {DATA_DIR}")
    
    # Show existing documents
    if DATA_DIR.exists():
        files = list(DATA_DIR.glob("*"))
        if files:
            st.subheader("Existing Documents")
            for file in files:
                st.write(f"✓ {file.name}")
        else:
            st.warning("No documents found in the data directory")
    
    # File uploader
    st.subheader("Upload New Documents")
    uploaded_files = st.file_uploader(
        "Choose files to upload",
        type=["pdf", "txt", "md"],
        accept_multiple_files=True
    )
    
    if uploaded_files:
        DATA_DIR.mkdir(exist_ok=True)
        
        for uploaded_file in uploaded_files:
            file_path = DATA_DIR / uploaded_file.name
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            st.success(f"✓ Uploaded: {uploaded_file.name}")
    
    # Build index button
    st.subheader("Build Index")
    if st.button("🔨 Build Vector Index", use_container_width=True):
        with st.spinner("Building index... This may take a few moments"):
            try:
                build_index()
                st.success("✓ Index built successfully!")
                st.session_state.vectordb_loaded = False  # Reset to reload
            except Exception as e:
                st.error(f"Error building index: {e}")


def page_rag_qa():
    """RAG Question Answering page"""
    st.title("❓ Question Answering")
    
    st.markdown("""
    Ask questions about your documents. The system will retrieve relevant 
    passages and generate answers based on the content.
    """)
    
    # Check if vector database exists
    if not CHROMA_DB_DIR.exists():
        st.warning("⚠️ No vector database found. Please upload documents and build the index first.")
        return
    
    # Load QA chain
    if not load_qa_chain():
        return
    
    # Query input
    query = st.text_input(
        "Enter your question:",
        placeholder="What are the main topics covered in the documents?"
    )
    
    if query and st.button("🔍 Get Answer", use_container_width=True):
        with st.spinner("Searching and generating answer..."):
            try:
                # Get answer from QA chain
                answer = st.session_state.qa_chain.invoke(query)
                
                # Get source documents
                vectordb = get_vectordb()
                retriever = vectordb.as_retriever(
                    search_type="similarity",
                    search_kwargs={"k": 3}
                )
                source_docs = retriever.invoke(query)
                
                # Display results
                st.subheader("Answer")
                st.write(answer)
                
                # Display sources
                st.subheader("Source Documents")
                for i, doc in enumerate(source_docs, 1):
                    with st.expander(f"Source {i}: {doc.metadata.get('source', 'Unknown')}"):
                        st.write(doc.page_content)
                
                # Add to conversation history
                st.session_state.conversation_history.append({
                    "query": query,
                    "answer": answer
                })
                
            except Exception as e:
                st.error(f"Error generating answer: {e}")
    
    # Display conversation history
    if st.session_state.conversation_history:
        st.subheader("Conversation History")
        for i, exchange in enumerate(st.session_state.conversation_history, 1):
            with st.expander(f"Q{i}: {exchange['query'][:50]}..."):
                st.write(f"**Q:** {exchange['query']}")
                st.write(f"**A:** {exchange['answer']}")


def page_multi_agent():
    """Multi-Agent Research page"""
    st.title("🤖 Multi-Agent Research Assistant")
    
    st.markdown("""
    This page uses a multi-agent system that:
    1. **Query Analysis** - Refines your query for better retrieval
    2. **Research** - Searches the vector database for relevant information
    3. **Synthesis** - Combines findings into a comprehensive answer
    """)
    
    # Check if vector database exists
    if not CHROMA_DB_DIR.exists():
        st.warning("⚠️ No vector database found. Please upload documents and build the index first.")
        return
    
    # Research query input
    research_query = st.text_area(
        "Enter your research question:",
        placeholder="What are the key insights and main findings in the documents?",
        height=100
    )
    
    if research_query and st.button("🔬 Run Research", use_container_width=True):
        with st.spinner("Running multi-agent research... This may take a moment"):
            try:
                result = run_research_assistant(research_query)
                
                # Display agent workflow
                st.subheader("Agent Workflow")
                for msg in result["agent_messages"]:
                    st.info(msg)
                
                # Display refined query
                st.subheader("Refined Query")
                st.write(result["refined_query"])
                
                # Display final answer
                st.subheader("Research Findings")
                st.write(result["final_answer"])
                
                # Display sources
                st.subheader("Source Documents")
                for i, source in enumerate(result["sources"], 1):
                    with st.expander(f"Source {i}: {source['source']}"):
                        st.write(source['content'])
                
            except Exception as e:
                st.error(f"Error running research: {e}")


def main():
    """Main app"""
    st.set_page_config(
        page_title="Research Assistant",
        page_icon="📚",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.sidebar.title("📚 Research Assistant")
    
    # Navigation
    page = st.sidebar.radio(
        "Select Mode:",
        options=["📄 Ingestion", "❓ RAG QA", "🤖 Multi-Agent Research"],
        key="page_radio"
    )
    
    # Initialize session state
    initialize_session_state()
    
    # Render selected page
    if page == "📄 Ingestion":
        page_ingestion()
    elif page == "❓ RAG QA":
        page_rag_qa()
    elif page == "🤖 Multi-Agent Research":
        page_multi_agent()
    
    # Sidebar info
    st.sidebar.markdown("---")
    st.sidebar.markdown("""
    ### About
    This is a research assistant powered by:
    - 🤗 HuggingFace Embeddings
    - 🔗 LangChain
    - 🦙 Ollama LLM
    - 🗄️ Chroma Vector DB
    - 📊 LangGraph Multi-Agent
    - 🎨 Streamlit UI
    """)


if __name__ == "__main__":
    main()
