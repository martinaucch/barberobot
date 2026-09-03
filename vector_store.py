import os
from dotenv import load_dotenv
import chromadb
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.mistralai import MistralAIEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core.ingestion import IngestionPipeline, DocstoreStrategy
from llama_index.core.storage.docstore import SimpleDocumentStore

# Import ingestion function
from ingestion import load_data_from_barberotheca

PERSIST_DIR = "./storage"

def update_vector_store():
    load_dotenv()
    # Fetch documents
    _, documents = load_data_from_barberotheca()
    
    if not documents:
        return

    # Initialize MistralAI embedding model
    embed_model = MistralAIEmbedding(model_name="mistral-embed")

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
    
    import time
    
    print(f"Processing {len(documents)} documents one by one with retry logic...")
    for i, doc in enumerate(documents):
        print(f"Processing Document {i+1}/{len(documents)}: {doc.id_}")
        
        success = False
        while not success:
            try:
                pipeline.run(documents=[doc], show_progress=True)
                success = True
                # Persist progress immediately so we don't start over if interrupted
                pipeline.docstore.persist(docstore_path)
                # Add a small buffer sleep to avoid hitting the limit constantly
                time.sleep(2)
            except Exception as e:
                err_msg = str(e)
                if any(x in err_msg for x in ["429", "Rate limit", "503", "temporarily unavailable", "high load"]):
                    print(f"API overload on document {doc.id_}. Waiting 30 seconds before retrying...")
                    time.sleep(30)
                else:
                    # If it's a different error, we should raise it to not get stuck in an infinite loop
                    raise e
    
    pipeline.docstore.persist(docstore_path)
    print(f"Done. Total chunks: {chroma_collection.count()}")

if __name__ == "__main__":
    # Create the storage directory if it doesn't exist
    os.makedirs(PERSIST_DIR, exist_ok=True)
    update_vector_store()
