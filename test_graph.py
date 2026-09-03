import sys
from graph_store import load_or_update_graph, CustomRDFRetriever

kg = load_or_update_graph()
retriever = CustomRDFRetriever(kg)

print("Entities loaded:", len(retriever.authoritative_entities))

# Let's check a few file IDs
for file_id in ["barbero-2018-FdM-2-Caporetto", "barbero-2012-FdM-3-Giovanna_d_Arco"]:
    print(f"\n--- Metadata for {file_id} ---")
    meta = retriever.get_lesson_metadata(file_id)
    print(meta)

