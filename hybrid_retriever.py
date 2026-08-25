import os
import chromadb
from llama_index.core import QueryBundle, Settings
from llama_index.core.retrievers import QueryFusionRetriever
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import VectorStoreIndex
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

# We are only testing retrieval, so we disable the default OpenAI LLM requirement
Settings.llm = None

# Import our custom components
from graph_store import load_or_update_graph, CustomRDFRetriever

PERSIST_DIR = "./storage"

def setup_hybrid_retriever():
    # Initializes both the Vector Database (ChromaDB) and the Knowledge Graph,
    # and fuses them together into a single, powerful Hybrid GraphRAG retriever.

    # ==========================================
    # 1. SETUP THE VECTOR RETRIEVER (ChromaDB)
    # ==========================================
    if not os.path.exists(PERSIST_DIR):
        raise FileNotFoundError("Storage directory not found. Please run vector_store.py first!")

    # A. Re-initialize the specific embedding model we used
    embed_model = HuggingFaceEmbedding(
        model_name="intfloat/multilingual-e5-large-instruct",
        query_instruction="Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery: ",
        text_instruction="" 
    )

    # B. Connect to ChromaDB
    db = chromadb.PersistentClient(path=PERSIST_DIR)
    chroma_collection = db.get_collection("barbero_e5_large")
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    
    # C. Build the LlamaIndex vector retriever
    vector_index = VectorStoreIndex.from_vector_store(
        vector_store, 
        embed_model=embed_model
    )
    # Fetch the top 3 most semantically similar text chunks
    vector_retriever = vector_index.as_retriever(similarity_top_k=3)

    # ==========================================
    # 2. SETUP THE GRAPH RETRIEVER (RDFlib)
    # ==========================================
    kg = load_or_update_graph()
    if not kg:
        raise ValueError("Failed to load the knowledge graph. Check ingestion.")
        
    # Our custom retriever that performs 1-hop BFS traversal
    graph_retriever = CustomRDFRetriever(rdf_graph=kg)

    # ==========================================
    # 3. FUSE THEM TOGETHER
    # ==========================================
    # QueryFusionRetriever executes the Parallel Query Processing and Dual-Path Retrieval
    hybrid_retriever = QueryFusionRetriever(
        retrievers=[vector_retriever, graph_retriever],
        # Retrieve the top 3 from Vector and the context from Graph, 
        # then return the top 4 combined nodes after scoring.
        similarity_top_k=4, 
        num_queries=1,
        # RRF (Reciprocal Rank Fusion) prioritizes nodes that score highly across BOTH databases
        mode="reciprocal_rerank" 
    )
    
    return hybrid_retriever

if __name__ == "__main__":
    # Test the Hybrid Retriever standalone
    retriever = setup_hybrid_retriever()
    
    test_query = "What did Barbero say about the Battle of Lepanto?"
    
    # Execute the fused retrieval
    nodes = retriever.retrieve(QueryBundle(test_query))
