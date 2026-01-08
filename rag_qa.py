from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
import time

"""
HuggingFaceEmbeddings - it imports the HuggingFaceEmbeddings class from langchain_huggingface to create embeddings using a specified model.
Chroma - it lets to interact with the chroma vector database from langchain_chroma package.
ChatOllama - it imports the ChatOllama class from langchain_ollama to interact with the Ollama LLM model.
ChatPromptTemplate - used to create prompts for the LLM
RunnablePassthrough - passes through the input to the next step in the chain
StrOutputParser - parses the LLM output as a string
"""

from config import CHROMA_DB_DIR, EMBEDDING_MODEL_NAME, OLLAMA_MODEL_NAME, RETRIEVER_TOP_K

def get_vectordb() -> Chroma:
    """
    Load the chroma vectordb
    it uses the same embedding model as used during the ingestion
    -> Chroma - means that it returns an instance of the Chroma vector database
    """
    embeddings=HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME
    )
    vectordb=Chroma(
        embedding_function=embeddings,
        persist_directory=str(CHROMA_DB_DIR)
    )
    """
    Chroma - creates an instance of the Chroma vector database
    embeddings - uses the same embedding model as used during the ingestion to ensure consistency
    persist_directory - directory where the vector database is stored in the disk.

    embeded query text uses the same embedding model to convert the query into vector 
    its because the vector database was created using the same embedding model
    if different embedding model is used then the vectors will not match and the retriver will break.
    """
    return vectordb

def get_qa_chain():
    """
    Create the RetrievalQA chain using LCEL (LangChain Expression Language)
    1. Loads the chroma vectordb
    2. retrieves the relevant chunks from the vectordb
    3. sends those chunks to the LLM model to get the final answer
    """
    vectordb=get_vectordb()

    retriever=vectordb.as_retriever(
        search_type="similarity",
        search_kwargs={"k":RETRIEVER_TOP_K}
    )
    """
    as_retriever - converts the vectordb into a retriever object which is used in langchain
    similarity - it retrieves the data based on vector similarity
    search_kwargs - these are additional parameters for the search function
    k - number of top results to return
    """

    llm=ChatOllama(
        model=OLLAMA_MODEL_NAME,
        temperature=0.1
    )
    """
    LLM is local and free of use
    Here the Ollama3 model is used
    temperature=0.1 - means that lower the temperature more deterministic and less creative
    """

    # Create a prompt template for the QA chain
    prompt = ChatPromptTemplate.from_template("""
    Use the following pieces of context to answer the question at the end.
    If you don't know the answer, just say that you don't know, don't try to make up an answer.
    
    Context: {context}
    
    Question: {input}
    
    Answer:
    """)
    
    """
    ChatPromptTemplate - creates a prompt template for the LLM
    The template includes placeholders for context (retrieved documents) and input (user question)
    """
    
    def format_docs(docs):
        """Format the retrieved documents into a single string"""
        return "\n\n".join(doc.page_content for doc in docs)
    
    # Create the retrieval chain using LCEL
    qa_chain = (
        {
            "context": retriever | format_docs,
            "input": RunnablePassthrough()
        }
        | prompt
        | llm
        | StrOutputParser()
    )
    """
    LCEL chain that:
    1. Takes input question via RunnablePassthrough
    2. Retrieves relevant documents using the retriever
    3. Formats documents into a single context string
    4. Combines context and question in the prompt
    5. Sends to LLM
    6. Parses the output as a string
    """
    
    return qa_chain

if __name__=="__main__":
    chain=get_qa_chain()
    vectordb=get_vectordb()
    retriever=vectordb.as_retriever(
        search_type="similarity",
        search_kwargs={"k":RETRIEVER_TOP_K}
    )
    
    query="Summarize key points from the documents."
    answer=chain.invoke(query)
    
    # Get source documents separately
    source_docs=retriever.invoke(query)

    print("=== Answer ===")
    print(answer)
    print("\n=== Source Documents ===")
    for doc in source_docs:
        source = doc.metadata.get('source', 'Unknown')
        print("-", source, "| snippet:", doc.page_content[:100])
