import os
from pathlib import Path
from typing import List

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

"""
PyPDFLoader - its used to load the pdf documents and convert them into langchain documents and metadata
TextLoader - its used to load the text documents and convert them into langchain documents and metadata
RecursiveCharacterTextSplitter - its used to split the documents into smaller chunks for better processing
HuggingFaceEmbeddings - its used to create embeddings using the specified model from langchain_huggingface package
Chroma - its used to create and manage the vector database from langchain_chroma package
"""
from config import DATA_DIR, CHROMA_DB_DIR, EMBEDDING_MODEL_NAME
"""
DATA_DIR - Directory where the pdf files are stored
CHROMA_DB_DIR - Directory where the vector database will be stored
EMBEDDING_MODEL_NAME - Name of the embedding model to be used
"""

def load_documents() -> List:
    """
    Load all PDF documents from the directory DATA_DIR
    and return a list of langchain documents.
    """
    docs=[]
    DATA_DIR.mkdir(exist_ok=True) # this checks whether the data directory exists or not and if does not exist it creates one
    for fname in os.listdir(DATA_DIR):
        path=DATA_DIR/fname
        # creates a path for each of the file which is present in the data directory
        if fname.lower().endswith(".pdf"):
            loader=PyPDFLoader(str(path))
            # if the files is in pdf format then it uses the PyPDFLoader to load the document and the PyPDFLoader expects the input as a string so str(path) converts the path into string.
        elif fname.lower().endswith(".txt") or fname.lower().endswith(".md"):
            loader=TextLoader(str(path))
            # if the files is in text or markdown format then it uses the TextLoader to load the document and the TextLoader expects the input as a string so str(path) converts the path into string.
        else:
            print(f"skipping {path} as it is not a supported file format.")
            continue
            # if the file is not in pdf or text or md format then it skips that file and continues to the next file.
        docs.extend(loader.load())
        """
        loader.load() loads and reads the document and returns a list of document objects
        for pdf - one document object for each page
        for text/md - one document object for the whole file
        docs.extend() adds the document to the list of docs
        """
    print(f"loaded {len(docs)} documents from {DATA_DIR}")
    return docs
    # return the list of documents to whichever function calls load_documents()

def build_index():
    """
    Build the vector index from the documents
    Pipeline:
    1. Load documents
    2. Split documents into smaller chunks
    3. Create embeddings
    4. Store embeddings in Chroma vector database
    """
    docs=load_documents()
    if not docs:
        print(f"no documents found. please add some documents to {DATA_DIR} and try again.")
        return
    # this loop checks whether the docs list is empty or not and returns if it is empty

    splitter=RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=200
    )
    """
    RecursiveCharacterTextSplitter - splits the documents into smaller chunks for better readability and processing
    chunk_size - number of character in each chunk(roughly 800 characters)
    chunk_overlap - number of overlapping characters between chunks(roughly 200 characters) for context of previous chunk
    """

    chunks=splitter.split_documents(docs)
    print(f"split into {len(chunks)} chunks")
    """
    chunks is a list which contains all the smaller chunks of the document objects in it
    it takes the list of documents as input and returns a list of smaller chunks of documents"""

    # create embeddings
    embeddings=HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
    """
    HuggingFaceEmbeddings - creates embeddings using the specified model
    model - comes from the config file it uses the sentence-transformers/all-MiniLM-L6-v2 model by default
    """

    # create vector database
    CHROMA_DB_DIR.mkdir(exist_ok=True)

    vectordb=Chroma.from_documents(
        documents=chunks,
        embedding_function=embeddings,
        persist_directory=str(CHROMA_DB_DIR)
    )
    """
    Chroma.from_documents - creates a Chroma vector database from the documents
    1. takes the chunks as input documents
    2. takes the embeddings object to create embeddings for each of the chunks
    3. persist_directory - directory where the vector database will be stored
    4. Creates a chroma vector database internally and it stores vector embeddings + original chunk + metadata
    Note: persist_directory automatically persists the database to disk means saves the database to disk
    """
    print(f"index built and stored at {CHROMA_DB_DIR}")

if __name__=="__main__":
    build_index()
    """
    this is called as script entry point it calls the build_index function to start the process of building the index
    this acts as the main function in java/c++
    """
      



