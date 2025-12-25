from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

from config import CHROMA_DB_DIR, EMBEDDING_MODEL_NAME

def view_data():
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)

    vectordb = Chroma(
        persist_directory=str(CHROMA_DB_DIR),
        embedding_function=embeddings
    )

    print("\n=== List of All IDs in Chroma ===")
    ids = vectordb.get()['ids']
    print(ids)

    print("\n=== Sample Documents ===")
    docs = vectordb.get()
    for i, doc in enumerate(docs["documents"]):
        print(f"\n--- Document {i+1} ---")
        print(doc[:500])  # print first 500 chars
        print("Metadata:", docs["metadatas"][i])

if __name__ == "__main__":
    view_data()
