import os
import re
import chromadb
from typing import List
from llama_index.core import QueryBundle
from llama_index.core.retrievers import BaseRetriever
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import VectorStoreIndex
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from FlagEmbedding import FlagReranker

# Import our custom components
from graph_store import load_or_update_graph, CustomRDFRetriever

PERSIST_DIR = "./storage"

# Regex patterns to strip conversational fluff from queries before embedding.
# This ensures the embedding model sees only the semantic core.
FLUFF_PATTERNS = [
    r"(?i)^ciao\s*(barberobot)?\s*[!.,]*\s*",
    r"(?i)^(buongiorno|buonasera|salve|hey|hi|hello)\s*[!.,]*\s*",
    r"(?i)^(cosa mi sai dire su(lle?|gli?|i)?|parlami di|raccontami di|dimmi qualcosa su(lle?|gli?|i)?|sai qualcosa su(lle?|gli?|i)?)\s*",
    r"(?i)^(cosa sai di|che cosa sai di|che mi dici di)\s*",
]

def clean_query(raw_query: str) -> str:
    """Strip greetings and meta-question phrasing, returning the semantic core."""
    cleaned = raw_query.strip()
    for pattern in FLUFF_PATTERNS:
        cleaned = re.sub(pattern, "", cleaned).strip()
    # If we accidentally stripped everything, fall back to the original
    return cleaned if len(cleaned) > 2 else raw_query.strip()


class SmartHybridRetriever(BaseRetriever):
    """
    Hybrid Retriever with Cross-Encoder Re-Ranking.
    
    Pipeline:
    1. Clean the query (strip conversational fluff)
    2. Run Graph Retrieval (inject entity facts as bonus context)
    3. Broad vector search (top 25 candidates)
    4. Cross-encoder re-rank all 25 candidates
    5. Return top 8 re-ranked chunks + graph facts
    """
    def __init__(self, vector_index: VectorStoreIndex, graph_retriever: CustomRDFRetriever, reranker: FlagReranker):
        self.vector_index = vector_index
        self.graph_retriever = graph_retriever
        self.reranker = reranker
        super().__init__()

    def _retrieve(self, query_bundle: QueryBundle) -> List:
        raw_query = query_bundle.query_str
        cleaned_query = clean_query(raw_query)
        
        # Use the cleaned query for both graph and vector retrieval
        effective_bundle = QueryBundle(cleaned_query)
        
        # 1. Graph Retrieval — inject entity facts as bonus context
        graph_nodes = self.graph_retriever._retrieve(effective_bundle)
        
        # 2. Broad vector search — retrieve a wide pool of candidates
        broad_retriever = self.vector_index.as_retriever(similarity_top_k=25)
        vector_nodes = broad_retriever.retrieve(effective_bundle)
        
        # 3. Deduplicate before re-ranking
        unique_nodes = []
        seen_text = set()
        for node in vector_nodes:
            text_snippet = node.node.get_text()[:100]
            if text_snippet not in seen_text:
                seen_text.add(text_snippet)
                unique_nodes.append(node)
        
        # 4. Cross-encoder re-ranking
        if unique_nodes:
            pairs = [[cleaned_query, node.node.get_text()] for node in unique_nodes]
            scores = self.reranker.compute_score(pairs, normalize=True)
            
            # Handle single result (returns float instead of list)
            if isinstance(scores, float):
                scores = [scores]
            
            # Sort by re-ranker score (descending)
            scored_nodes = list(zip(scores, unique_nodes))
            scored_nodes.sort(key=lambda x: x[0], reverse=True)
            
            # Take top 8 after re-ranking
            reranked_nodes = [node for _, node in scored_nodes[:8]]
        else:
            reranked_nodes = []
        
        return graph_nodes + reranked_nodes


def setup_hybrid_retriever():
    if not os.path.exists(PERSIST_DIR):
        raise FileNotFoundError("Storage directory not found. Please run vector_store.py first!")

    embed_model = HuggingFaceEmbedding(
        model_name="intfloat/multilingual-e5-large-instruct",
        query_instruction="Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery: ",
        text_instruction="",
        device="cpu"
    )
    from llama_index.core import Settings
    Settings.embed_model = embed_model

    db = chromadb.PersistentClient(path=PERSIST_DIR)
    chroma_collection = db.get_collection("barbero_e5_large")
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    
    vector_index = VectorStoreIndex.from_vector_store(
        vector_store, 
        embed_model=embed_model
    )
    
    kg = load_or_update_graph()
    if not kg:
        raise ValueError("Failed to load the knowledge graph.")
        
    graph_retriever = CustomRDFRetriever(rdf_graph=kg)
    
    # Initialize the BGE cross-encoder re-ranker
    # Forcing fp32/cpu to prevent Mac Apple Silicon (MPS) silent hangs
    reranker = FlagReranker("BAAI/bge-reranker-v2-m3", use_fp16=False)
    
    smart_retriever = SmartHybridRetriever(vector_index, graph_retriever, reranker)
    return smart_retriever


if __name__ == "__main__":
    from llama_index.core import Settings
    Settings.llm = None
    
    retriever = setup_hybrid_retriever()
    
    print("\n--- TEST 1: Specific Entity (Carducci) ---")
    nodes1 = retriever.retrieve(QueryBundle("Cosa sai di Carducci?"))
    for n in nodes1:
        print(n.node.get_text()[:150].replace('\n', ' '))
        
    print("\n--- TEST 2: Conceptual (Malattie nel medioevo) ---")
    nodes2 = retriever.retrieve(QueryBundle("ciao barberobot! cosa mi sai dire sulle malattie nel medioevo?"))
    for n in nodes2:
        meta = n.node.metadata
        file_id = meta.get("file_id", "graph")
        print(f"  [{file_id}] {n.node.get_text()[:120].replace(chr(10), ' ')}")
