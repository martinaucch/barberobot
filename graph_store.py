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

    def extract_entities(self, query_text: str) -> set[URIRef]:
        """Extract entities from the user query."""
        user_query = query_text.lower()
        matched_nodes = set()
        
        sorted_entities = sorted(self.authoritative_entities.keys(), key=len, reverse=True)
        
        for surface_form in sorted_entities:
            if re.search(rf'\b{re.escape(surface_form)}\b', user_query):
                rdf_name = self.authoritative_entities[surface_form]
                exact_uri = URIRef(f"{self.namespace}{rdf_name}")
                matched_nodes.add(exact_uri)
                user_query = re.sub(rf'\b{re.escape(surface_form)}\b', '', user_query)
                
        return matched_nodes

    def get_lessons_for_entities(self, entities: set[URIRef]) -> list[str]:
        """Finds lesson file_ids that reference any of the given entities."""
        from rdflib.namespace import DCTERMS
        lesson_ids = set()
        for entity in entities:
            # Find subjects where predicate is DCTERMS.references and object is entity
            for s in self.rdf_graph.subjects(DCTERMS.references, entity):
                # s is a lesson URI, get its identifier (which maps to file_id)
                for identifier in self.rdf_graph.objects(s, DCTERMS.identifier):
                    lesson_ids.add(str(identifier))
        return list(lesson_ids)

    def get_lesson_metadata(self, file_id: str) -> str:
        """Retrieves formatted metadata for a specific lesson file_id."""
        from rdflib.namespace import DCTERMS
        from rdflib import URIRef
        
        lesson_uri = None
        for s, p, o in self.rdf_graph.triples((None, DCTERMS.identifier, None)):
            if str(o) == file_id:
                lesson_uri = s
                break
                
        if not lesson_uri:
            return ""
            
        title = next(self.rdf_graph.objects(lesson_uri, DCTERMS.title), "Titolo Sconosciuto")
        source = next(self.rdf_graph.objects(lesson_uri, DCTERMS.source), "")
        main_entity = next(self.rdf_graph.objects(lesson_uri, URIRef("https://schema.org/mainEntityOfPage")), "")
        
        is_part_of = next(self.rdf_graph.objects(lesson_uri, DCTERMS.isPartOf), None)
        series_info = ""
        if is_part_of:
            series_name = next(self.rdf_graph.objects(is_part_of, URIRef("https://schema.org/name")), "")
            series_title = next(self.rdf_graph.objects(is_part_of, DCTERMS.title), "")
            series_date = next(self.rdf_graph.objects(is_part_of, DCTERMS.date), "")
            
            # Format nicely
            if series_title:
                series_info = f"Questa lezione fa parte della serie '{series_name}: {series_title}' ({series_date})"
            else:
                series_info = f"Questa lezione fa parte della serie '{series_name}' ({series_date})"
            
        metadata_str = f"Metadata for lesson '{file_id}':\n"
        metadata_str += f"- Titolo Lezione: {title}\n"
        if source:
            metadata_str += f"- Guarda su YouTube: {source}\n"
        if main_entity:
            metadata_str += f"- Trascrizione completa: {main_entity}\n"
        if series_info:
            metadata_str += f"- Consigli correlati: {series_info}\n"
            
        return metadata_str

    def _retrieve(self, query_bundle: QueryBundle):
        # We don't use this standard LlamaIndex interface anymore in our hybrid flow.
        return []

if __name__ == "__main__":
    kg = load_or_update_graph()
    if kg:
        retriever = CustomRDFRetriever(kg)
        test_query = "What did Barbero say about Giosuè Carducci?"
        results = retriever._retrieve(QueryBundle(test_query))
        for r in results:
            print(r.node.text)
