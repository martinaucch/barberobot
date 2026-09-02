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
        self._build_entity_indices()
        super().__init__()

    def _build_entity_indices(self):
        """Precomputes entity<->lesson reverse indices, AND a file_id->lesson_uri
        index, in a single pass over the graph. Done once at startup (not
        per-query) so lookups afterwards are dict lookups instead of repeated
        full-graph triple scans (get_lesson_metadata used to do
        `for s,p,o in self.rdf_graph.triples(...)` on EVERY call, up to 8x per
        query - that's an O(n_lessons) scan repeated needlessly)."""
        from rdflib.namespace import DCTERMS
        from collections import defaultdict

        entity_to_lessons = defaultdict(set)
        lesson_to_entities = defaultdict(set)
        file_id_to_lesson_uri = {}

        for lesson_uri, _, entity_uri in self.rdf_graph.triples((None, DCTERMS.references, None)):
            entity_to_lessons[entity_uri].add(lesson_uri)
            lesson_to_entities[lesson_uri].add(entity_uri)

        for lesson_uri, _, identifier in self.rdf_graph.triples((None, DCTERMS.identifier, None)):
            file_id_to_lesson_uri[str(identifier)] = lesson_uri

        self.entity_to_lessons = entity_to_lessons
        self.lesson_to_entities = lesson_to_entities
        self.file_id_to_lesson_uri = file_id_to_lesson_uri

    def _load_authoritative_entities(self):
        """Loads the authoritative CSV into a dictionary for exact Named Entity Recognition."""
        entities_dict = {}
        original_names = {}
        if os.path.exists(ENTITIES_CSV_PATH):
            with open(ENTITIES_CSV_PATH, mode='r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    # Map the lowercase human string to the exact RDF name
                    # e.g., "giovanna d'arco" -> "GiovannaDArco"
                    surface_form = row['entity'].lower().strip()
                    rdf_name = row['rdf_name'].strip()
                    entities_dict[surface_form] = rdf_name
                    if rdf_name not in original_names:
                        original_names[rdf_name] = row['entity'].strip()
        self.original_entity_names = original_names
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

    def get_related_lessons_by_entities(self, lesson_uri: URIRef, min_shared: int = 2, min_score: float = 0.30, max_results: int = 5) -> list[tuple[str, list[str]]]:
        """Finds other lessons that share at least `min_shared` entities with the
        given lesson. Ranks by an inverse-frequency-weighted score, so a shared
        entity that appears in only a couple of lessons (e.g. 'Firenze') counts far
        more than one that appears everywhere (e.g. 'Dio', 'Chiesa') - the raw
        min_shared count is only the qualifying threshold, not the final ranking.
        Returns a list of tuples: (lesson_title, [shared_entity_names])."""
        from collections import defaultdict

        this_lesson_entities = self.lesson_to_entities.get(lesson_uri, set())
        if not this_lesson_entities:
            return []

        shared_count = defaultdict(int)
        weighted_score = defaultdict(float)
        shared_entities = defaultdict(list)

        for entity_uri in this_lesson_entities:
            lessons_with_entity = self.entity_to_lessons.get(entity_uri, set())
            if not lessons_with_entity:
                continue
            # Rarer entities (fewer lessons referencing them) score higher.
            weight = 1.0 / len(lessons_with_entity)
            for other_lesson in lessons_with_entity:
                if other_lesson == lesson_uri:
                    continue
                shared_count[other_lesson] += 1
                weighted_score[other_lesson] += weight
                shared_entities[other_lesson].append(entity_uri)

        qualifying = [
            (lesson, weighted_score[lesson], shared_entities[lesson])
            for lesson, count in shared_count.items()
            if count >= min_shared and weighted_score[lesson] >= min_score
        ]
        qualifying.sort(key=lambda x: x[1], reverse=True)

        from rdflib.namespace import DCTERMS
        results = []
        for other_lesson_uri, _, ents in qualifying[:max_results]:
            title = next(self.rdf_graph.objects(other_lesson_uri, DCTERMS.title), None)
            if title:
                # Convert entity URIs back to human readable names
                ent_names = []
                for e in ents:
                    rdf_name = str(e).split('/')[-1]
                    human_name = self.original_entity_names.get(rdf_name, rdf_name)
                    ent_names.append(human_name)
                results.append((str(title), ent_names))
        return results

    def get_lesson_metadata(self, file_id: str) -> str:
        """Retrieves formatted metadata for a specific lesson file_id."""
        from rdflib.namespace import DCTERMS
        from rdflib import URIRef

        # O(1) lookup via the prebuilt index instead of scanning every
        # dcterms:identifier triple in the graph on every call.
        lesson_uri = self.file_id_to_lesson_uri.get(file_id)

        if not lesson_uri:
            return ""
            
        title = next(self.rdf_graph.objects(lesson_uri, DCTERMS.title), "Titolo Sconosciuto")
        source = next(self.rdf_graph.objects(lesson_uri, DCTERMS.source), "")
        main_entity = next(self.rdf_graph.objects(lesson_uri, URIRef("https://schema.org/mainEntityOfPage")), "")
        
        is_part_of = next(self.rdf_graph.objects(lesson_uri, DCTERMS.isPartOf), None)
        series_info = ""
        sibling_lessons = []
        if is_part_of:
            series_name = next(self.rdf_graph.objects(is_part_of, URIRef("https://schema.org/name")), "")
            series_title = next(self.rdf_graph.objects(is_part_of, DCTERMS.title), "")
            series_date = next(self.rdf_graph.objects(is_part_of, DCTERMS.date), "")
            
            # Find sibling lessons in the same series
            sibling_lessons = []
            for sibling_uri in self.rdf_graph.subjects(DCTERMS.isPartOf, is_part_of):
                if sibling_uri != lesson_uri:
                    sibling_title = next(self.rdf_graph.objects(sibling_uri, DCTERMS.title), None)
                    if sibling_title:
                        sibling_lessons.append(str(sibling_title))
                        
            # Format nicely
            if series_title:
                series_info = f"Questa lezione fa parte della serie '{series_name}: {series_title}' ({series_date})"
            else:
                series_info = f"Questa lezione fa parte della serie '{series_name}' ({series_date})"
                
            if sibling_lessons:
                series_info += f". Altre lezioni in questa serie: {', '.join(sibling_lessons)}"

        # Entity-based related lessons (independent of series/event membership).
        # Filtered against sibling_lessons so we don't recommend the same lesson twice.
        entity_related = self.get_related_lessons_by_entities(lesson_uri, min_shared=2, max_results=5)
        entity_related_filtered = [r for r in entity_related if r[0] not in sibling_lessons]

        metadata_str = f"Metadata for lesson '{file_id}':\n"
        metadata_str += f"- Titolo Lezione: {title}\n"
        if source:
            metadata_str += f"- Guarda su YouTube: {source}\n"
        if main_entity:
            metadata_str += f"- Trascrizione completa: {main_entity}\n"
        if series_info:
            metadata_str += f"- Consigli correlati: {series_info}\n"
        if entity_related_filtered:
            related_strs = []
            for t, ents in entity_related_filtered:
                related_strs.append(f"{t} (in comune: {', '.join(ents)})")
            metadata_str += f"- Altre lezioni con temi/entità in comune: {'; '.join(related_strs)}\n"

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