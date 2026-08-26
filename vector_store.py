import os
import chromadb
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core.ingestion import IngestionPipeline, DocstoreStrategy
from llama_index.core.storage.docstore import SimpleDocumentStore

# Import ingestion function
from ingestion import load_data_from_barberotheca

PERSIST_DIR = "./storage"

def update_vector_store():
    # Fetch documents
    _, documents = load_data_from_barberotheca()
    
    if not documents:
        return

    # Initialize HuggingFace embedding model for multilingual-e5-large-instruct
    embed_model = HuggingFaceEmbedding(
        model_name="intfloat/multilingual-e5-large-instruct",
        query_instruction="Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery: ",
        text_instruction="",
        device="cpu"
    )

    # Set up ChromaDB Vector Database
    db = chromadb.PersistentClient(path=PERSIST_DIR)
    chroma_collection = db.get_or_create_collection("barbero_e5_large")
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    
    # Set up the Document Store to track which files are vectorized
    docstore_path = os.path.join(PERSIST_DIR, "docstore.json")
    if os.path.exists(docstore_path):
        # Load the history of previously embedded files
        docstore = SimpleDocumentStore.from_persist_path(docstore_path)
    else:
        # First time running the script
        docstore = SimpleDocumentStore()
        
    # Build the Ingestion Pipeline
    pipeline = IngestionPipeline(
        transformations=[
            SentenceSplitter(chunk_size=256, chunk_overlap=64),
            embed_model,
        ],
        docstore=docstore,
        docstore_strategy=DocstoreStrategy.UPSERTS,
        vector_store=vector_store,
    )
    
    # Process documents one at a time to avoid MPS memory pressure
    for i, doc in enumerate(documents):
        print(f"[{i+1}/{len(documents)}] {doc.id_}")
        pipeline.run(documents=[doc], show_progress=True)
    
    pipeline.docstore.persist(docstore_path)
    print(f"Done. Total chunks: {chroma_collection.count()}")

if __name__ == "__main__":
    # Create the storage directory if it doesn't exist
    os.makedirs(PERSIST_DIR, exist_ok=True)
    update_vector_store()
