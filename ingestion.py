import pandas as pd
from llama_index.core import Document
from pathlib import Path
import subprocess

def ensure_submodule_data(base_path: Path):
    """Ensures the submodule is initialized and sparse-checkout is configured."""
    transcripts_path = base_path / "transcripts"
    
    if not transcripts_path.exists() or not any(transcripts_path.iterdir()):
        print("Barberotheca transcripts not detected. Configuring sparse-checkout automatically...")
        subprocess.run(["git", "submodule", "update", "--init", "--depth", "1"], check=True)
        subprocess.run(["git", "-C", str(base_path), "sparse-checkout", "init", "--cone"], check=True)
        subprocess.run(["git", "-C", str(base_path), "sparse-checkout", "set", "transcripts", "metadata"], check=True)
        print("Data successfully pulled and filtered!")

def load_data_from_barberotheca(base_dir="data/barberotheca"):
    """
    Loads metadata and transcript text files from the barberotheca partial clone.
    Returns a Pandas DataFrame of the metadata and a list of LlamaIndex Documents.
    """
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
            
    print(f"Successfully loaded {len(documents)} transcripts and metadata mapping.")
    return barbero_df, documents

if __name__ == "__main__":
    metadata_df, docs = load_data_from_barberotheca()
