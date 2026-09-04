import os
import re
import chromadb
from typing import List
from llama_index.core import QueryBundle
from llama_index.core.retrievers import BaseRetriever
from llama_index.core.vector_stores import MetadataFilters, MetadataFilter, VectorStoreQuery
from llama_index.core.schema import NodeWithScore
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import VectorStoreIndex
from llama_index.embeddings.cohere import CohereEmbedding
from llama_index.postprocessor.cohere_rerank import CohereRerank

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
    def __init__(self, vector_index: VectorStoreIndex, graph_retriever: CustomRDFRetriever, reranker: CohereRerank):
        self.vector_index = vector_index
        self.graph_retriever = graph_retriever
        self.reranker = reranker
        super().__init__()

    def _retrieve(self, query_bundle: QueryBundle) -> List:
        raw_query = query_bundle.query_str
        cleaned_query = clean_query(raw_query)
        
        # Use the cleaned query for both graph and vector retrieval
        effective_bundle = QueryBundle(cleaned_query)
        
        # 1. Entity Extraction & Lesson Disambiguation via Knowledge Graph
        entities = self.graph_retriever.extract_entities(cleaned_query)
        entity_lesson_ids = set(self.graph_retriever.get_lessons_for_entities(entities)) if entities else set()

        boosted_lesson_ids = entity_lesson_ids

        # 2. Compute the query embedding once to avoid redundant API calls during broad and forced retrievals.
        query_embedding = self.vector_index._embed_model.get_query_embedding(cleaned_query)

        def _vector_query(top_k: int, filters=None):
            vs_query = VectorStoreQuery(
                query_embedding=query_embedding,
                similarity_top_k=top_k,
                filters=filters,
            )
            result = self.vector_index.vector_store.query(vs_query)
            similarities = result.similarities or [0.0] * len(result.nodes)
            return [NodeWithScore(node=n, score=s) for n, s in zip(result.nodes, similarities)]

        # Broad vector search — retrieve a wide pool of candidates
        vector_nodes = _vector_query(top_k=8)

        # Forced retrieval: explicitly fetch top chunks from KG-identified lessons to ensure they enter the reranking pool.
        forced_nodes = []
        for file_id in boosted_lesson_ids:
            filters = MetadataFilters(filters=[MetadataFilter(key="file_id", value=file_id)])
            forced_nodes.extend(_vector_query(top_k=3, filters=filters))

        all_candidates = vector_nodes + forced_nodes

        # 3. Deduplicate candidates
        unique_nodes = []
        seen_text = set()
        for node in all_candidates:
            text_snippet = node.node.get_text()[:100]
            if text_snippet not in seen_text:
                seen_text.add(text_snippet)
                unique_nodes.append(node)
        
        # 4. Cross-encoder re-ranking with boosting
        if unique_nodes:
            self.reranker.top_n = len(unique_nodes) # Ensure it scores all candidates
            cohere_nodes = self.reranker.postprocess_nodes(unique_nodes, QueryBundle(cleaned_query))
            
            # Boost scores for chunks belonging to graph-identified lessons, applying a floor of 0.5.
            boosted_scores = []
            for node in cohere_nodes:
                score = node.score or 0.0
                file_id = node.node.metadata.get("file_id")
                if file_id in boosted_lesson_ids:
                    boosted_scores.append(max(score + 0.5, 0.5))
                else:
                    boosted_scores.append(score)
            
            # Sort by boosted re-ranker score (descending)
            scored_nodes = list(zip(boosted_scores, cohere_nodes))
            scored_nodes.sort(key=lambda x: x[0], reverse=True)
            
            # Take top 8 after re-ranking
            reranked_nodes = [node for _, node in scored_nodes[:8]]
            
            for boosted_score, node in scored_nodes[:8]:
                node.score = boosted_score
        else:
            reranked_nodes = []

        # 5. Graph Metadata Enrichment
        from llama_index.core.schema import TextNode
        metadata_nodes = []
        seen_file_ids = set()
        
        for node in reranked_nodes:
            file_id = node.node.metadata.get("file_id")
            if file_id and file_id not in seen_file_ids:
                seen_file_ids.add(file_id)
                metadata_str = self.graph_retriever.get_lesson_metadata(file_id)
                if metadata_str:
                    text_node = TextNode(
                        text=metadata_str,
                        metadata={"source": "rdflib_graph_metadata", "file_id": file_id}
                    )
                    metadata_nodes.append(NodeWithScore(node=text_node, score=1.0))
        
        # We return the metadata text blocks first, so the LLM reads them before the chunks
        return metadata_nodes + reranked_nodes


def setup_hybrid_retriever():
    if not os.path.exists(PERSIST_DIR):
        raise FileNotFoundError("Storage directory not found. Please run vector_store.py first!")

    embed_model = CohereEmbedding(
        model_name="embed-multilingual-v3.0", 
        input_type="search_query",
        cohere_api_key=os.environ.get("COHERE_API_KEY")
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
    
    # Initialize Cohere Reranker
    reranker = CohereRerank(
        model="rerank-multilingual-v3.0", 
        top_n=50,
        api_key=os.environ.get("COHERE_API_KEY")
    )
    
    smart_retriever = SmartHybridRetriever(vector_index, graph_retriever, reranker)
    return smart_retriever