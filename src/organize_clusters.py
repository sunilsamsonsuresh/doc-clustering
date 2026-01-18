from pathlib import Path
import shutil
import pandas as pd
from tqdm import tqdm

from .config import Config

def organize(cfg: Config, mode: str = "copy"):
    """
    mode: 'copy' (safe) or 'move'
    """
    cfg.clusters_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(cfg.clusters_csv)
    for row in tqdm(df.itertuples(index=False), total=len(df), desc="Organizing PDFs"):
        pdf_path = Path(row.pdf_path)
        cluster_id = row.cluster_id
        folder = "noise" if cluster_id == -1 else f"cluster_{cluster_id}"
        out_dir = cfg.clusters_dir / folder
        out_dir.mkdir(parents=True, exist_ok=True)

        dest = out_dir / pdf_path.name
        if mode == "move":
            shutil.move(str(pdf_path), str(dest))
        else:
            shutil.copy2(str(pdf_path), str(dest))

    print(f"✅ Cluster folders created at: {cfg.clusters_dir}")

if __name__ == "__main__":
    organize(Config(), mode="copy")
