from typing import Any, Dict, List, Union
from langgraph.graph import StateGraph, START, END
from langchain_ollama import ChatOllama
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from langchain_core.output_parsers import JsonOutputParser
from config import CHROMA_DB_DIR, EMBEDDING_MODEL_NAME, OLLAMA_MODEL_NAME, RETRIEVER_TOP_K

"""
Multi-Agent Research Assistant Graph
This implements a multi-agent system using LangGraph for:
1. Query Analysis Agent - Analyzes and refines user queries
2. Research Agent - Retrieves relevant information from vector database
3. Synthesis Agent - Synthesizes final answer from research findings
"""

class AgentState(BaseModel):
    """State structure for the multi-agent system"""
    query: str = Field(description="Original user query")
    refined_query: str = Field(default="", description="Refined query for better retrieval")
    retrieved_docs: List[Dict] = Field(default_factory=list, description="Retrieved documents")
    research_findings: str = Field(default="", description="Compiled research findings")
    final_answer: str = Field(default="", description="Final synthesized answer")
    messages: List[str] = Field(default_factory=list, description="Agent messages")


def get_llm():
    """Initialize the LLM"""
    return ChatOllama(
        model=OLLAMA_MODEL_NAME,
        temperature=0.1
    )


def get_vectordb() -> Chroma:
    """Load the chroma vectordb"""
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
    vectordb = Chroma(
        embedding_function=embeddings,
        persist_directory=str(CHROMA_DB_DIR)
    )
    return vectordb


def query_analysis_agent(state: AgentState) -> Dict[str, Any]:
    """
    Agent 1: Analyze and refine the user query
    This agent takes the original query and refines it for better retrieval
    """
    llm = get_llm()
    
    prompt = ChatPromptTemplate.from_template("""
    You are a query analysis expert. Your task is to refine and enhance the user's query 
    to make it more suitable for document retrieval.
    
    Original Query: {query}
    
    Please provide a refined version of the query that:
    1. Is clear and specific
    2. Includes key concepts
    3. Removes ambiguity
    
    Respond with just the refined query, nothing else.
    """)
    
    chain = prompt | llm
    refined_query = chain.invoke({"query": state.query}).content.strip()
    
    state.refined_query = refined_query
    state.messages.append(f"Query Analysis Agent: Refined query to - '{refined_query}'")
    
    return {
        "refined_query": refined_query,
        "messages": state.messages
    }


def research_agent(state: AgentState) -> Dict[str, Any]:
    """
    Agent 2: Retrieve relevant documents from the vector database
    This agent performs semantic search to find relevant chunks
    """
    vectordb = get_vectordb()
    
    retriever = vectordb.as_retriever(
        search_type="similarity",
        search_kwargs={"k": RETRIEVER_TOP_K}
    )
    
    # Use refined query if available, otherwise use original query
    query_to_use = state.refined_query if state.refined_query else state.query
    
    # Retrieve documents
    docs = retriever.invoke(query_to_use)
    
    # Format retrieved documents
    retrieved_docs = []
    for i, doc in enumerate(docs):
        doc_dict = {
            "id": i,
            "source": doc.metadata.get("source", "Unknown"),
            "content": doc.page_content
        }
        retrieved_docs.append(doc_dict)
    
    # Compile research findings
    findings = "\n\n".join([
        f"Source: {doc['source']}\nContent: {doc['content']}" 
        for doc in retrieved_docs
    ])
    
    state.research_findings = findings
    state.retrieved_docs = retrieved_docs
    state.messages.append(f"Research Agent: Retrieved {len(docs)} relevant documents")
    
    return {
        "retrieved_docs": retrieved_docs,
        "research_findings": findings,
        "messages": state.messages
    }


def synthesis_agent(state: AgentState) -> Dict[str, Any]:
    """
    Agent 3: Synthesize findings into a comprehensive answer
    This agent generates the final answer based on retrieved information
    """
    llm = get_llm()
    
    prompt = ChatPromptTemplate.from_template("""
    You are an expert research synthesizer. Your task is to provide a comprehensive answer 
    based on the research findings provided.
    
    Original Question: {query}
    
    Research Findings:
    {findings}
    
    Please provide a well-structured answer that:
    1. Directly addresses the query
    2. Incorporates information from the research findings
    3. Is clear and concise
    4. If you don't know the answer, say so
    """)
    
    chain = prompt | llm
    
    final_answer = chain.invoke({
        "query": state.query,
        "findings": state.research_findings
    }).content.strip()
    
    state.final_answer = final_answer
    state.messages.append("Synthesis Agent: Generated final answer")
    
    return {
        "final_answer": final_answer,
        "messages": state.messages
    }


def create_research_graph():
    """Create the multi-agent graph"""
    graph = StateGraph(AgentState)
    
    # Add nodes
    graph.add_node("query_analysis", query_analysis_agent)
    graph.add_node("research", research_agent)
    graph.add_node("synthesis", synthesis_agent)
    
    # Add edges - sequential execution
    graph.add_edge(START, "query_analysis")
    graph.add_edge("query_analysis", "research")
    graph.add_edge("research", "synthesis")
    graph.add_edge("synthesis", END)
    
    # Compile the graph
    compiled_graph = graph.compile()
    
    return compiled_graph


def run_research_assistant(query: str) -> Dict[str, Any]:
    """
    Run the multi-agent research assistant
    
    Args:
        query: User's research query
    
    Returns:
        Dictionary containing the final answer and agent messages
    """
    graph = create_research_graph()
    
    initial_state = AgentState(query=query)
    
    result = graph.invoke(initial_state)
    
    return {
        "query": result.get("query", ""),
        "refined_query": result.get("refined_query", ""),
        "final_answer": result.get("final_answer", ""),
        "sources": result.get("retrieved_docs", []),
        "agent_messages": result.get("messages", [])
    }


if __name__ == "__main__":
    # Test the multi-agent system
    test_query = "What are the main topics covered in the documents?"
    
    print(f"\n{'='*60}")
    print(f"Research Query: {test_query}")
    print(f"{'='*60}")
    
    result = run_research_assistant(test_query)
    
    print("\n=== Agent Messages ===")
    for msg in result["agent_messages"]:
        print(f"• {msg}")
    
    print(f"\n=== Refined Query ===")
    print(result["refined_query"])
    
    print(f"\n=== Final Answer ===")
    print(result["final_answer"])
    
    print(f"\n=== Source Documents ===")
    for i, source in enumerate(result["sources"], 1):
        print(f"\n{i}. Source: {source['source']}")
        print(f"   Snippet: {source['content'][:200]}...")
