import pandas as pd
from llama_index.core import Document
from pathlib import Path
import subprocess

# Ensures the submodule is initialized and sparse-checkout is configured.
def ensure_submodule_data(base_path: Path):
    transcripts_path = base_path / "transcripts"
    
    if not transcripts_path.exists() or not any(transcripts_path.iterdir()):
        subprocess.run(["git", "submodule", "update", "--init", "--depth", "1"], check=True)
        subprocess.run(["git", "-C", str(base_path), "sparse-checkout", "init", "--cone"], check=True)
        subprocess.run(["git", "-C", str(base_path), "sparse-checkout", "set", "transcripts", "metadata"], check=True)

# Loads metadata and transcript text files from the barberotheca partial clone.
# Returns a Pandas DataFrame of the metadata and a list of LlamaIndex Documents.
def load_data_from_barberotheca(base_dir="data/barberotheca"):
    base_path = Path(base_dir)
    ensure_submodule_data(base_path)
    
    barbero_df = pd.read_csv(base_path / "metadata" / "barbero.csv")
    documents = []
    
    for txt_path in (base_path / "transcripts").glob("*.txt"):
        with open(txt_path, "r", encoding="utf-8") as f:
            documents.append(Document(
                text=f.read(), 
                metadata={"source": str(txt_path), "file_id": txt_path.stem}
            ))
            
    return barbero_df, documents

if __name__ == "__main__":
    metadata_df, docs = load_data_from_barberotheca()
