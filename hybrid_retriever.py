import os
import re
import time
import chromadb
from typing import List
from llama_index.core import QueryBundle
from llama_index.core.retrievers import BaseRetriever
from llama_index.core.vector_stores import MetadataFilters, MetadataFilter, VectorStoreQuery
from llama_index.core.schema import NodeWithScore
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import VectorStoreIndex
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from sentence_transformers import CrossEncoder
import torch

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
    def __init__(self, vector_index: VectorStoreIndex, graph_retriever: CustomRDFRetriever, reranker: CrossEncoder):
        self.vector_index = vector_index
        self.graph_retriever = graph_retriever
        self.reranker = reranker
        super().__init__()

    def _retrieve(self, query_bundle: QueryBundle) -> List:
        t0 = time.time()
        raw_query = query_bundle.query_str
        cleaned_query = clean_query(raw_query)
        
        # Use the cleaned query for both graph and vector retrieval
        effective_bundle = QueryBundle(cleaned_query)
        
        # 1. Entity Extraction & Lesson Disambiguation via Knowledge Graph
        entities = self.graph_retriever.extract_entities(cleaned_query)
        entity_lesson_ids = set(self.graph_retriever.get_lessons_for_entities(entities)) if entities else set()

        boosted_lesson_ids = entity_lesson_ids
        t1 = time.time()
        print(f"[TIMING] entity extraction: {t1 - t0:.2f}s")

        # 2. Compute the query embedding ONCE and reuse it for every vector call
        # below. Previously, each `.as_retriever().retrieve()` call silently
        # re-embedded the same query text with a heavy CPU embedding model
        # (multilingual-e5-large-instruct, ~560M params) - once for the broad
        # search, then AGAIN for every single boosted lesson in the forced-
        # retrieval loop. With N matched lessons that's N+1 embedding
        # computations for one query. Embedding once and passing the vector
        # directly removes all that redundant work.
        query_embedding = self.vector_index._embed_model.get_query_embedding(cleaned_query)
        t2 = time.time()
        print(f"[TIMING] query embedding (x1): {t2 - t1:.2f}s")

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
        vector_nodes = _vector_query(top_k=12)
        t3 = time.time()
        print(f"[TIMING] broad vector search: {t3 - t2:.2f}s")

        # GUARANTEED RETRIEVAL for KG-identified lessons.
        # Without this, boosted_lesson_ids can point at a lesson that has ZERO
        # chunks in the top-25 pool (e.g. an entity mentioned in passing in a
        # lesson that isn't semantically "about" it - this was the Carducci bug).
        # A boost with nothing to boost does nothing, so we explicitly fetch the
        # best-matching chunk(s) from each identified lesson via a metadata filter,
        # guaranteeing they're at least in the pool to be reranked and boosted.
        forced_nodes = []
        for file_id in boosted_lesson_ids:
            filters = MetadataFilters(filters=[MetadataFilter(key="file_id", value=file_id)])
            forced_nodes.extend(_vector_query(top_k=3, filters=filters))
        t4 = time.time()
        print(f"[TIMING] forced retrieval ({len(boosted_lesson_ids)} lessons): {t4 - t3:.2f}s")

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
            pairs = [[cleaned_query, node.node.get_text()] for node in unique_nodes]
            scores = self.reranker.predict(pairs, batch_size=len(pairs), activation_function=torch.nn.Sigmoid())
            t5 = time.time()
            print(f"[TIMING] cross-encoder rerank ({len(pairs)} pairs): {t5 - t4:.2f}s")
            
            scores = scores.tolist()
            
            # Boost scores for chunks that belong to the graph-disambiguated lessons.
            # The floor (max with 0.5) matters for force-fetched nodes: a chunk that
            # only entered the pool via the metadata filter above may get a low raw
            # cross-encoder score (it wasn't semantically top-matched to begin with),
            # so the additive boost alone might not be enough to keep it in the top 8.
            boosted_scores = []
            for score, node in zip(scores, unique_nodes):
                file_id = node.node.metadata.get("file_id")
                if file_id in boosted_lesson_ids:
                    boosted_scores.append(max(score + 0.5, 0.5))
                else:
                    boosted_scores.append(score)
            
            # Sort by boosted re-ranker score (descending)
            scored_nodes = list(zip(boosted_scores, unique_nodes))
            scored_nodes.sort(key=lambda x: x[0], reverse=True)
            
            # Take top 8 after re-ranking
            reranked_nodes = [node for _, node in scored_nodes[:8]]
        else:
            reranked_nodes = []
            t5 = time.time()

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
        t6 = time.time()
        print(f"[TIMING] graph metadata enrichment: {t6 - t5:.2f}s")
        print(f"[TIMING] TOTAL _retrieve(): {t6 - t0:.2f}s")
        
        # We return the metadata text blocks first, so the LLM reads them before the chunks
        return metadata_nodes + reranked_nodes


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
    
    # Initialize the BGE cross-encoder re-ranker.
    # Using the smaller "base" variant instead of "v2-m3": v2-m3 (~568M params) is
    # built for maximum multilingual quality, but on a CPU-only deployment (no GPU)
    # its cost dominates total latency (measured: ~0.6-0.7s per query-chunk pair).
    # bge-reranker-base is roughly half the parameters and still multilingual
    # (XLM-RoBERTa-base backbone), trading a bit of ranking precision for a
    # meaningful, direct cut in per-query latency on CPU.
    # fp16 is left disabled on purpose: on CPU (no CUDA/MPS) fp16 generally does
    # NOT speed up inference and can even slow it down due to missing optimized
    # kernels - it's a GPU/MPS optimization, not a CPU one.
    reranker = CrossEncoder("BAAI/bge-reranker-base", max_length=320)
    
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