## Synthetic Insurance Document Clustering Demo (DiT+ HDBSCAN)

## How to Contribute:

### 1) Install
```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# Mac/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

---

# 📄 Document Clustering & Classification Lab

This repository is an experimental framework for **vision-based document understanding**, focused on:

* 📦 Unsupervised document clustering
* 🧠 Layout-driven embeddings (no OCR required)
* 🏗 Template and document-family discovery
* 🏥 Insurance-style document intake use cases (FNOL, claims, policies, receipts, correspondence)

It supports multiple state-of-the-art document vision models and a modular pipeline to go from **raw documents → embeddings → clusters → organized folders**.

---

# 🎯 What this project does

✔ Convert mixed documents into standardized first-page images
✔ Embed documents using modern vision/document models
✔ Cluster documents using UMAP + HDBSCAN
✔ Organize documents into cluster folders
✔ Evaluate clustering quality when labels are available
✔ Serve as a foundation for building document classifiers

---

# 🧠 Supported models

The framework currently supports:

| Model                                | Type                 | OCR | Best for                                   |
| ------------------------------------ | -------------------- | --- | ------------------------------------------ |
| **DiT (Document Image Transformer)** | Document vision      | ❌   | **Primary model for layout clustering**    |
| DINOv2 (small/base/large)            | Foundation vision    | ❌   | Strong general layout & template discovery |
| LayoutLMv3                           | Multimodal doc model | ✅   | Layout + text-aware embeddings             |
| Donut (encoder only)                 | Doc understanding    | ❌   | Heavy but document-specialized             |

➡ In practice, **DiT gave the strongest and most stable clustering results** for large unlabeled document sets.

---

# 🏗 Repository structure

```
doc-clustering/
├── data/
│   ├── <source_folders>/        # invoice/, form/, fnol/, receipts/, etc.
│   ├── page1_pngs/               # normalized first-page images
│   ├── raw_pdfs/                 # wrapped PDFs for cluster delivery
│   └── meta.csv                  # document registry
│
├── outputs/
│   ├── embeddings_*.npy
│   ├── umap_*.npy
│   ├── clusters_*.csv
│   └── clusters/                 # cluster folders with PDFs
│
├── src/
│   ├── prepare_local_rvl.py      # dataset preparation
│   ├── cluster_dinov2.py
│   ├── cluster_layoutlmv3.py
│   ├── cluster_donut.py
│   ├── cluster_dit.py            # ⭐ recommended
│   ├── organize_clusters.py
│   └── config.py
│
└── README.md
```

---

# 🔄 End-to-end pipeline

## 1️⃣ Prepare dataset (ingestion layer)

Organize your raw documents like:

```
data/
  fnol/
  claim_form/
  invoice/
  receipt/
  correspondence/
```

Then run:

```bash
python3 -m src.prepare_local_rvl
```

This step:

* samples documents
* converts first pages to PNG
* wraps images into single-page PDFs
* builds `meta.csv`

Outputs:

```
data/page1_pngs/
data/raw_pdfs/
data/meta.csv
```

This layer decouples **ingestion** from **ML models**.

---

## 2️⃣ Generate embeddings + clusters

### ⭐ DiT (recommended)

```bash
python3 -m src.cluster_dit
```

### DINOv2

```bash
python3 -m src.cluster_dinov2
```

### LayoutLMv3 (OCR + layout)

```bash
python3 -m src.cluster_layoutlmv3
```

### Donut encoder

```bash
python3 -m src.cluster_donut
```

Each script:

* embeds first-page images
* reduces dimensionality with UMAP
* clusters using HDBSCAN
* writes cluster labels to `outputs/clusters_*.csv`

---

## 3️⃣ Organize documents into cluster folders

```bash
python3 -m src.organize_clusters
```

Creates:

```
outputs/clusters/
  cluster_0/
  cluster_1/
  cluster_2/
```

Each folder contains **PDFs of documents belonging to that cluster** — ideal for:

* visual inspection
* stakeholder demos
* manual labeling
* downstream automation

---

# 🏥 Insurance-style use case

This framework is designed for scenarios such as:

* FNOL form discovery
* Claim document grouping
* Policy vs invoice vs correspondence separation
* Template family detection
* Intake triage and automation

Typical flow:

```
Unlabeled documents
   ↓
DiT embeddings
   ↓
HDBSCAN clusters
   ↓
Human review & naming
   ↓
Training set creation
   ↓
Supervised document classifier
```

This turns unsupervised discovery into a **production-grade document classification system**.

---

# 📊 Evaluation

If `meta.csv` contains `doc_type`, clustering scripts automatically report:

* ARI (Adjusted Rand Index)
* NMI (Normalized Mutual Information)
* number of clusters
* noise ratio

For truly unlabeled corpora, evaluation is visual and cluster-purity driven.

---

# ⚙ Configuration

All core settings live in `src/config.py`:

```python
# Models
dinov2_model = "facebook/dinov2-large"
dit_model    = "microsoft/dit-base-finetuned-rvlcdip"

# Embedding
batch_size = 16
device = "auto"

# UMAP
umap_n_components = 15
umap_n_neighbors = 35
umap_min_dist = 0.1

# HDBSCAN
hdb_min_cluster_size = 30
hdb_min_samples = 8
```

---

# 🚀 Recommended setup for large unlabeled corpora

For 3k–10k documents:

```
Model: DiT
UMAP neighbors: 30–50
Min cluster size: 30–60
Min samples: 6–12
```

---

# 🧩 Why this architecture

* Vision-first (no OCR dependency)
* Template and layout sensitive
* Scales to thousands of documents
* Model-agnostic embedding layer
* Business-friendly cluster outputs
* Natural bridge to supervised classification

---

# 🛣 Roadmap ideas

* cluster thumbnails & HTML browser
* hybrid vision + OCR embeddings
* reclustering large clusters
* active-learning loop
* classifier training pipeline
* Databricks / cloud batch mode

---

# 📌 Key takeaway

> This project is not just clustering — it is a **document understanding pipeline** for discovering, structuring, and operationalizing large unlabeled document collections.
