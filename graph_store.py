import os
import hashlib
import pickle
import re
import csv
import requests
from pathlib import Path
from rdflib import Graph, URIRef
from llama_index.core.retrievers import BaseRetriever
from llama_index.core.schema import NodeWithScore, TextNode
from llama_index.core import QueryBundle

# Paths
BASE_DIR = "data/barberotheca/metadata"
TTL_FILE_PATH = os.path.join(BASE_DIR, "knowledge-graph.ttl")
ENTITIES_CSV_PATH = os.path.join(BASE_DIR, "entities-authoritative.csv")

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
    def __init__(self, rdf_graph: Graph):
        self.rdf_graph = rdf_graph
        self.namespace = "https://github.com/metamuses/barberotheca/entity/"
        self.authoritative_entities = self._load_authoritative_entities()
        super().__init__()

    def _load_authoritative_entities(self):
        """Loads the authoritative CSV into a dictionary for exact Named Entity Recognition."""
        entities_dict = {}
        if os.path.exists(ENTITIES_CSV_PATH):
            with open(ENTITIES_CSV_PATH, mode='r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    # Map the lowercase human string to the exact RDF name
                    # e.g., "giovanna d'arco" -> "GiovannaDArco"
                    surface_form = row['entity'].lower().strip()
                    rdf_name = row['rdf_name'].strip()
                    entities_dict[surface_form] = rdf_name
        return entities_dict

    def _fetch_wikidata_summary(self, wikidata_url: str):
        """Fetches a live description from Wikidata using the Q-ID!"""
        try:
            q_id = wikidata_url.split("/")[-1]
            api_url = f"https://www.wikidata.org/w/api.php?action=wbgetentities&ids={q_id}&format=json&props=descriptions&languages=it"
            headers = {"User-Agent": "BarberoBot/1.0 (barberobot@example.com)"}
            response = requests.get(api_url, headers=headers, timeout=2).json()
            description = response['entities'][q_id]['descriptions']['it']['value']
            return f" (Wikidata Live Description: {description})"
        except Exception:
            return ""

    def _format_node(self, node):
        if hasattr(node, "fragment") and node.fragment:
            return node.fragment
        if hasattr(node, "toPython") and str(node).startswith("http"):
            return str(node).split("/")[-1]
        return str(node)

    def _retrieve(self, query_bundle: QueryBundle):
        user_query = query_bundle.query_str.lower()
        matched_nodes = set()

        # =========================================================
        # STEP 1: AUTHORITATIVE ENTITY MATCHING (The Smart Way)
        # =========================================================
        # We check if any of our exact entities from the CSV are in the user's question
        for surface_form, rdf_name in self.authoritative_entities.items():
            if surface_form in user_query:
                # We found an exact match! (e.g. "giovanna d'arco")
                # Construct the exact URI used in the graph
                exact_uri = URIRef(f"{self.namespace}{rdf_name}")
                matched_nodes.add(exact_uri)

        # Fallback: If no exact entities matched, we fall back to keyword regex matching
        if not matched_nodes:
            stop_words = {"what", "who", "where", "is", "the", "a", "an", "of", "in", "about", "did", "say", "cosa", "chi", "come", "dove", "dice"}
            words = re.findall(r'\b\w+\b', user_query)
            keywords = [w for w in words if w not in stop_words and len(w) > 3]
            
            for subj, pred, obj in self.rdf_graph:
                subj_str, obj_str = str(subj).lower(), str(obj).lower()
                for kw in keywords:
                    if re.search(rf'\b{re.escape(kw)}\b', subj_str):
                        matched_nodes.add(subj)
                    if re.search(rf'\b{re.escape(kw)}\b', obj_str):
                        matched_nodes.add(obj)

        if not matched_nodes:
            return []

        # =========================================================
        # STEP 2: GRAPH TRAVERSAL + WIKIDATA API
        # =========================================================
        nodes_with_scores = []
        seen_triples = set()
        
        for node in matched_nodes:
            # Traversal A: OUTGOING
            for s, p, o in self.rdf_graph.triples((node, None, None)):
                wikidata_context = ""
                if "sameAs" in str(p) and "wikidata.org" in str(o):
                    wikidata_context = self._fetch_wikidata_summary(str(o))

                clean_s, clean_p, clean_o = self._format_node(s), self._format_node(p), self._format_node(o)
                triple_str = f"{clean_s} -> {clean_p} -> {clean_o}{wikidata_context}"
                
                if triple_str not in seen_triples:
                    seen_triples.add(triple_str)
                    text_node = TextNode(
                        text=f"Knowledge Graph Fact: {triple_str}",
                        metadata={"source": "rdflib_graph_traversal", "entity": str(node)}
                    )
                    nodes_with_scores.append(NodeWithScore(node=text_node, score=1.0))

            # Traversal B: INCOMING
            for s, p, o in self.rdf_graph.triples((None, None, node)):
                clean_s, clean_p, clean_o = self._format_node(s), self._format_node(p), self._format_node(o)
                triple_str = f"{clean_s} -> {clean_p} -> {clean_o}"
                
                if triple_str not in seen_triples:
                    seen_triples.add(triple_str)
                    text_node = TextNode(
                        text=f"Knowledge Graph Fact: {triple_str}",
                        metadata={"source": "rdflib_graph_traversal", "entity": str(node)}
                    )
                    nodes_with_scores.append(NodeWithScore(node=text_node, score=1.0))
        
        return nodes_with_scores

if __name__ == "__main__":
    kg = load_or_update_graph()
    if kg:
        retriever = CustomRDFRetriever(kg)
        test_query = "What did Barbero say about Giosuè Carducci?"
        results = retriever._retrieve(QueryBundle(test_query))
        for r in results:
            print(r.node.text)
