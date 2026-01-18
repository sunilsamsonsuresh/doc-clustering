from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm
from PIL import Image

import torch
import umap
import hdbscan
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

from transformers import DonutProcessor, VisionEncoderDecoderModel

from .config import Config


def resolve_device(cfg: Config) -> str:
    if getattr(cfg, "device", "auto") in ("cpu", "cuda", "mps"):
        d = cfg.device
        if d == "cuda" and not torch.cuda.is_available():
            return "cpu"
        if d == "mps" and not torch.backends.mps.is_available():
            return "cpu"
        return d
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


@torch.inference_mode()
def embed_donut_encoder(
    cfg: Config,
    image_paths: list[Path],
    model_name: str = "naver-clova-ix/donut-base",
) -> np.ndarray:
    """
    Donut is an encoder-decoder model. For clustering, we only use the *vision encoder*
    to get a fixed-size embedding per page.

    Embedding used: mean-pooled last_hidden_state of the vision encoder (B, seq, hidden).
    """
    device = resolve_device(cfg)

    processor = DonutProcessor.from_pretrained(model_name)
    model = VisionEncoderDecoderModel.from_pretrained(model_name).to(device)
    model.eval()

    # Batch size: Donut is heavier than DINO; keep it conservative on CPU/MPS
    if device == "cuda":
        bs = getattr(cfg, "batch_size", 16)
    elif device == "mps":
        bs = 1
    else:
        bs = 1


    X_parts: list[np.ndarray] = []

    for i in tqdm(range(0, len(image_paths), bs), desc="Donut encoder embedding"):
        batch_paths = image_paths[i : i + bs]
        imgs = [Image.open(p).convert("RGB") for p in batch_paths]

        pixel_values = processor(images=imgs, return_tensors="pt").pixel_values.to(device)

        # Run only the vision encoder
        enc_out = model.encoder(pixel_values=pixel_values, return_dict=True)
        hs = enc_out.last_hidden_state  # (B, seq, hidden)

        # Mean-pool across sequence dimension (patch tokens)
        emb = hs.mean(dim=1)  # (B, hidden)
        X_parts.append(emb.detach().cpu().float().numpy())

    return np.vstack(X_parts)


def cluster_donut(cfg: Config):
    cfg.outputs_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(cfg.meta_csv)

    # Prefer a direct image column; fall back to page1_png if present
    if "page1_path" in df.columns:
        image_paths = [Path(p) for p in df["page1_path"].tolist()]
    elif "page1_png" in df.columns:
        image_paths = [Path(p) for p in df["page1_png"].tolist()]
    else:
        raise RuntimeError("meta.csv needs a 'page1_path' or 'page1_png' column.")

    missing = [str(p) for p in image_paths if not p.exists()]
    if missing:
        raise FileNotFoundError(f"{len(missing)} image files missing. Example: {missing[0]}")

    model_name = getattr(cfg, "donut_model", "naver-clova-ix/donut-base")

    # 1) Embeddings
    X = embed_donut_encoder(cfg, image_paths, model_name=model_name)
    np.save(cfg.outputs_dir / "embeddings_donut.npy", X)

    # 2) UMAP
    reducer = umap.UMAP(
        n_components=cfg.umap_n_components,
        n_neighbors=cfg.umap_n_neighbors,
        min_dist=cfg.umap_min_dist,
        metric="cosine",
        random_state=cfg.seed,
    )
    Z = reducer.fit_transform(X)
    np.save(cfg.outputs_dir / "umap_donut.npy", Z)

    # 3) HDBSCAN
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=cfg.hdb_min_cluster_size,
        min_samples=cfg.hdb_min_samples,
        metric="euclidean",
    )
    labels = clusterer.fit_predict(Z)

    df_out = df.copy()
    df_out["cluster_id"] = labels
    out_csv = cfg.outputs_dir / "clusters_donut.csv"
    df_out.to_csv(out_csv, index=False)

    # Optional evaluation if you have doc_type
    mask = labels != -1
    ari = adjusted_rand_score(df_out.loc[mask, "doc_type"], df_out.loc[mask, "cluster_id"]) if mask.any() and "doc_type" in df_out.columns else 0.0
    nmi = normalized_mutual_info_score(df_out.loc[mask, "doc_type"], df_out.loc[mask, "cluster_id"]) if mask.any() and "doc_type" in df_out.columns else 0.0

    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    noise = float(np.mean(labels == -1))
    print(f"✅ Donut clusters={n_clusters}  noise={noise:.2%}  ARI={ari:.3f}  NMI={nmi:.3f}")
    print(f"✅ wrote {out_csv}")


if __name__ == "__main__":
    cluster_donut(Config())
