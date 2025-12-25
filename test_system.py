"""
Test script to verify the research assistant is working
This script tests all major components without requiring documents
"""
import sys
from pathlib import Path

def test_imports():
    """Test that all modules can be imported"""
    print("\n" + "="*60)
    print("TESTING IMPORTS")
    print("="*60)
    
    try:
        from ingestion_build_index import load_documents, build_index
        print("✓ ingestion_build_index imported successfully")
    except Exception as e:
        print(f"✗ Failed to import ingestion_build_index: {e}")
        return False
    
    try:
        from rag_qa import get_qa_chain, get_vectordb
        print("✓ rag_qa imported successfully")
    except Exception as e:
        print(f"✗ Failed to import rag_qa: {e}")
        return False
    
    try:
        from graph_multi_agent import run_research_assistant, create_research_graph
        print("✓ graph_multi_agent imported successfully")
    except Exception as e:
        print(f"✗ Failed to import graph_multi_agent: {e}")
        return False
    
    try:
        import app_streamlit
        print("✓ app_streamlit imported successfully")
    except Exception as e:
        print(f"✗ Failed to import app_streamlit: {e}")
        return False
    
    try:
        from config import DATA_DIR, CHROMA_DB_DIR, EMBEDDING_MODEL_NAME, OLLAMA_MODEL_NAME
        print("✓ config imported successfully")
    except Exception as e:
        print(f"✗ Failed to import config: {e}")
        return False
    
    return True


def test_configuration():
    """Test configuration settings"""
    print("\n" + "="*60)
    print("TESTING CONFIGURATION")
    print("="*60)
    
    from config import DATA_DIR, CHROMA_DB_DIR, EMBEDDING_MODEL_NAME, OLLAMA_MODEL_NAME, RETRIEVER_TOP_K
    
    print(f"✓ Data Directory: {DATA_DIR}")
    print(f"✓ Chroma DB Directory: {CHROMA_DB_DIR}")
    print(f"✓ Embedding Model: {EMBEDDING_MODEL_NAME}")
    print(f"✓ Ollama Model: {OLLAMA_MODEL_NAME}")
    print(f"✓ Retriever Top-K: {RETRIEVER_TOP_K}")
    
    # Check if directories exist
    if DATA_DIR.exists():
        print(f"✓ Data directory exists: {DATA_DIR}")
    else:
        print(f"! Data directory not found (will be created on ingestion): {DATA_DIR}")
    
    return True


def test_vector_db_exists():
    """Check if vector database exists"""
    print("\n" + "="*60)
    print("TESTING VECTOR DATABASE")
    print("="*60)
    
    from config import CHROMA_DB_DIR
    
    if CHROMA_DB_DIR.exists():
        print(f"✓ Vector database exists: {CHROMA_DB_DIR}")
        db_files = list(CHROMA_DB_DIR.glob("*"))
        if db_files:
            print(f"  Database contains {len(db_files)} items")
            return True
        else:
            print("! Vector database is empty - run ingestion first")
            return False
    else:
        print(f"! Vector database not found: {CHROMA_DB_DIR}")
        print("  Run 'python ingestion_build_index.py' to create it")
        return False


def test_graph_creation():
    """Test that we can create the research graph"""
    print("\n" + "="*60)
    print("TESTING GRAPH CREATION")
    print("="*60)
    
    try:
        from graph_multi_agent import create_research_graph
        graph = create_research_graph()
        print("✓ Research graph created successfully")
        return True
    except Exception as e:
        print(f"✗ Failed to create research graph: {e}")
        return False


def test_embeddings():
    """Test that embeddings can be loaded"""
    print("\n" + "="*60)
    print("TESTING EMBEDDINGS")
    print("="*60)
    
    try:
        from langchain_huggingface import HuggingFaceEmbeddings
        from config import EMBEDDING_MODEL_NAME
        
        print(f"Loading embeddings model: {EMBEDDING_MODEL_NAME}")
        embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
        print("✓ Embeddings model loaded successfully")
        
        # Test embedding a small text
        test_text = "This is a test document"
        embedding = embeddings.embed_query(test_text)
        print(f"✓ Embedding dimension: {len(embedding)}")
        
        return True
    except Exception as e:
        print(f"✗ Failed to load embeddings: {e}")
        return False


def main():
    """Run all tests"""
    print("\n")
    print("╔" + "="*58 + "╗")
    print("║" + " "*10 + "RESEARCH ASSISTANT TEST SUITE" + " "*19 + "║")
    print("╚" + "="*58 + "╝")
    
    tests = [
        ("Imports", test_imports),
        ("Configuration", test_configuration),
        ("Embeddings", test_embeddings),
        ("Graph Creation", test_graph_creation),
        ("Vector Database", test_vector_db_exists),
    ]
    
    results = {}
    for test_name, test_func in tests:
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"\n✗ Test '{test_name}' failed with exception: {e}")
            results[test_name] = False
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    
    for test_name, passed_test in results.items():
        status = "✓ PASS" if passed_test else "✗ FAIL"
        print(f"{status}: {test_name}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n✓ All tests passed! System is ready to use.")
        print("\nNext steps:")
        print("1. Add PDF/TXT/MD files to the 'data' directory")
        print("2. Run: python ingestion_build_index.py")
        print("3. Run: streamlit run app_streamlit.py")
    else:
        print(f"\n! {total - passed} test(s) failed. Please check the errors above.")
    
    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
