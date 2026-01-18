from dataclasses import dataclass
from pathlib import Path

@dataclass
class Config:
    n_docs: int = 500
    seed: int = 42

    data_dir: Path = Path("data")
    raw_pdf_dir: Path = Path("data/raw_pdfs")
    page1_img_dir: Path = Path("data/page1_pngs")

    outputs_dir: Path = Path("outputs")
    clusters_dir: Path = Path("outputs/clusters")

    meta_csv: Path = Path("data/meta.csv")
    clusters_csv: Path = Path("outputs/clusters.csv")

    dinov2_model: str = "facebook/dinov2-large"  # try dinov2-small for speed
    layoutlmv3_model: str = "microsoft/layoutlmv3-base"
    donut_model: str = "naver-clova-ix/donut-base"
    batch_size: int = 32
    device: str = "auto"  # "auto" | "cpu" | "cuda"

    # UMAP + HDBSCAN params (tweakable)
    umap_n_components: int = 15
    umap_n_neighbors: int = 35
    umap_min_dist: float = 0.15
    hdb_min_cluster_size: int = 10 # How many docs should a cluster have at least to be considered a cluster
    hdb_min_samples: int = 5 # How many samples in a neighborhood for a point to be considered a core point
