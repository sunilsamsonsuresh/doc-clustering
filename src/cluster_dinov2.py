from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm
from PIL import Image

import torch
from transformers import AutoImageProcessor, Dinov2Model

import umap
import hdbscan
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

from .config import Config

def resolve_device(cfg: Config) -> str:
    if cfg.device in ("cpu", "cuda"):
        return cfg.device
    return "cuda" if torch.cuda.is_available() else "cpu"

@torch.inference_mode()
def embed_images(cfg: Config, image_paths: list[Path]) -> np.ndarray:
    device = resolve_device(cfg)
    model_name = cfg.dinov2_model

    processor = AutoImageProcessor.from_pretrained(model_name)
    model = Dinov2Model.from_pretrained(model_name).to(device)
    model.eval()

    bs = cfg.batch_size if device == "cuda" else max(4, cfg.batch_size // 4)

    X_parts = []
    for i in tqdm(range(0, len(image_paths), bs), desc="DINOv2 embedding"):
        batch_paths = image_paths[i:i+bs]
        imgs = [Image.open(p).convert("RGB") for p in batch_paths]
        inputs = processor(images=imgs, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=(device == "cuda")):
            out = model(**inputs)
            emb = out.last_hidden_state[:, 0, :]  # CLS token

        X_parts.append(emb.detach().cpu().float().numpy())

    return np.vstack(X_parts)

def cluster(cfg: Config):
    cfg.outputs_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(cfg.meta_csv)
    image_paths = [Path(p) for p in df["page1_png"].tolist()]

    if "page1_png" in df.columns:
        image_paths = [Path(p) for p in df["page1_png"].tolist()]
    else:
        image_paths = [cfg.page1_img_dir / (Path(p).stem + ".png") for p in df["pdf_path"].tolist()]


    # 1) Embeddings
    X = embed_images(cfg, image_paths)
    np.save(cfg.outputs_dir / "embeddings.npy", X)

    # 2) UMAP (recommended before HDBSCAN)
    reducer = umap.UMAP(
        n_components=cfg.umap_n_components,
        n_neighbors=cfg.umap_n_neighbors,
        min_dist=cfg.umap_min_dist,
        metric="cosine",
        random_state=cfg.seed,
    )
    Z = reducer.fit_transform(X)
    np.save(cfg.outputs_dir / "umap.npy", Z)

    # 3) HDBSCAN
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=cfg.hdb_min_cluster_size,
        min_samples=cfg.hdb_min_samples,
        metric="euclidean",
    )
    labels = clusterer.fit_predict(Z)

    df_out = df.copy()
    df_out["cluster_id"] = labels
    df_out.to_csv(cfg.clusters_csv, index=False)

    # quick evaluation (since synthetic has ground truth)
    mask = labels != -1
    if mask.sum() > 0:
        ari = adjusted_rand_score(df_out.loc[mask, "doc_type"], df_out.loc[mask, "cluster_id"])
        nmi = normalized_mutual_info_score(df_out.loc[mask, "doc_type"], df_out.loc[mask, "cluster_id"])
    else:
        ari, nmi = 0.0, 0.0

    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    noise = float(np.mean(labels == -1))
    print(f"✅ clusters={n_clusters}  noise={noise:.2%}  ARI={ari:.3f}  NMI={nmi:.3f}")
    print(f"✅ wrote {cfg.clusters_csv}")

if __name__ == "__main__":
    cluster(Config())
