import sys
from graph_store import load_or_update_graph, CustomRDFRetriever

kg = load_or_update_graph()
retriever = CustomRDFRetriever(kg)

from rdflib.namespace import DCTERMS
count = 0
for lesson_uri in set(retriever.lesson_to_entities.keys()):
    identifier = str(next(retriever.rdf_graph.objects(lesson_uri, DCTERMS.identifier), ""))
    if not identifier: continue
    
    meta = retriever.get_lesson_metadata(identifier)
    if "Altre lezioni con temi/entità in comune:" in meta:
        print(f"File ID: {identifier}")
        # print the specific line
        for line in meta.split('\n'):
            if "Altre lezioni con temi/entità in comune:" in line:
                print("  " + line)
        count += 1

print(f"Total lessons with entity recommendations: {count}")
