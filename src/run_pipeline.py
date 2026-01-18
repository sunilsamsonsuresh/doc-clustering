from .config import Config
from .prepare_local_rvl import prepare_local_rvl
from .cluster_dinov2 import cluster
from .organize_clusters import organize

def main():
    cfg = Config(n_docs=1000)
    prepare_local_rvl(cfg)
    cluster(cfg)
    organize(cfg, mode="copy")

if __name__ == "__main__":
    main()
