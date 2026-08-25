import os
import hashlib
import pickle
import re
from pathlib import Path
from rdflib import Graph
from llama_index.core.retrievers import BaseRetriever
from llama_index.core.schema import NodeWithScore, TextNode
from llama_index.core import QueryBundle

# Paths
TTL_FILE_PATH = "data/barberotheca/metadata/knowledge-graph.ttl"
PERSIST_DIR = "./storage"
GRAPH_CACHE_PATH = os.path.join(PERSIST_DIR, "rdflib_graph.pkl")
GRAPH_HASH_PATH = os.path.join(PERSIST_DIR, "graph_hash.txt")

def get_file_hash(filepath):
    # Generates an MD5 hash of the file to detect if it has been updated.
    if not os.path.exists(filepath):
        return None
    with open(filepath, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()

def load_or_update_graph():
    # Loads the graph from cache if untouched, or parses the .ttl if it's new/updated.
    current_hash = get_file_hash(TTL_FILE_PATH)
    if not current_hash:
        return None

    if os.path.exists(GRAPH_HASH_PATH) and os.path.exists(GRAPH_CACHE_PATH):
        with open(GRAPH_HASH_PATH, "r") as f:
            saved_hash = f.read().strip()
            
        if saved_hash == current_hash:
            with open(GRAPH_CACHE_PATH, "rb") as f:
                return pickle.load(f)
                
    # New or updated .ttl file detected. Parsing triples into memory...
    g = Graph()
    g.parse(TTL_FILE_PATH, format="turtle")
    
    os.makedirs(PERSIST_DIR, exist_ok=True)
    with open(GRAPH_CACHE_PATH, "wb") as f:
        pickle.dump(g, f)
    with open(GRAPH_HASH_PATH, "w") as f:
        f.write(current_hash)
        
    return g

# ==========================================
# CUSTOM GRAPHRAG RETRIEVER
# ==========================================
class CustomRDFRetriever(BaseRetriever):
    # A GraphRAG retriever. It identifies starting nodes based on the query,
    # and then performs a 1-hop Breadth-First Search (BFS) traversal to retrieve 
    # connected structural context.
    
    def __init__(self, rdf_graph: Graph):
        self.rdf_graph = rdf_graph
        super().__init__()

    def _extract_keywords(self, query: str):
        # A simple helper to extract meaningful words from the query.
        # Remove common stop words for better entity matching
        stop_words = {"what", "who", "where", "is", "the", "a", "an", "of", "in", "about", "did", "say"}
        words = re.findall(r'\b\w+\b', query.lower())
        return [w for w in words if w not in stop_words and len(w) > 2]

    def _format_node(self, node):
        # Extracts the readable part of a URI, or returns the literal string.
        if hasattr(node, "fragment") and node.fragment:
            return node.fragment
        # If it's a URIRef but has no fragment, grab the last part of the path
        if hasattr(node, "toPython") and str(node).startswith("http"):
            return str(node).split("/")[-1]
        return str(node)

    def _retrieve(self, query_bundle: QueryBundle):
        user_query = query_bundle.query_str.lower()
        keywords = self._extract_keywords(user_query)
        
        if not keywords:
            return []

        # =========================================================
        # STEP 1: Find "Anchor Nodes" (Entity Matching)
        # =========================================================
        matched_nodes = set()
        for subj, pred, obj in self.rdf_graph:
            subj_str = str(subj).lower()
            obj_str = str(obj).lower()
            
            # If any keyword is in the Subject or Object, it's an anchor node
            for kw in keywords:
                if kw in subj_str:
                    matched_nodes.add(subj)
                if kw in obj_str:
                    matched_nodes.add(obj)

        if not matched_nodes:
            return []

        # =========================================================
        # STEP 2: Graph Traversal (1-Hop BFS)
        # =========================================================
        relevant_triples = set()
        
        for node in matched_nodes:
            # Traversal A: Find all OUTGOING relationships (node -> pred -> obj)
            for s, p, o in self.rdf_graph.triples((node, None, None)):
                clean_s = self._format_node(s)
                clean_p = self._format_node(p)
                clean_o = self._format_node(o)
                relevant_triples.add(f"{clean_s} -> {clean_p} -> {clean_o}")

            # Traversal B: Find all INCOMING relationships (subj -> pred -> node)
            for s, p, o in self.rdf_graph.triples((None, None, node)):
                clean_s = self._format_node(s)
                clean_p = self._format_node(p)
                clean_o = self._format_node(o)
                relevant_triples.add(f"{clean_s} -> {clean_p} -> {clean_o}")

        # =========================================================
        # STEP 3: Package Context for the LLM
        # =========================================================
        combined_graph_context = "\n".join(relevant_triples)
        
        node = TextNode(
            text=f"Knowledge Graph Context (1-Hop Traversal):\n{combined_graph_context}",
            metadata={"source": "rdflib_graph_traversal"}
        )
        
        return [NodeWithScore(node=node, score=1.0)]

if __name__ == "__main__":
    # Test the standalone GraphRAG Traversal
    kg = load_or_update_graph()
    if kg:
        retriever = CustomRDFRetriever(kg)
        test_query = "Where is Lesson 42 set?"
        
        results = retriever._retrieve(QueryBundle(test_query))
