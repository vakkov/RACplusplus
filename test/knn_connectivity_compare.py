#!/usr/bin/env python3
import argparse
import os
import sys
import time
from typing import Optional, Tuple

import numpy as np
import scipy.sparse as sp
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.neighbors import kneighbors_graph

# Allow running from a source checkout without installing the package.
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(REPO_ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)


def build_knn_connectivity(
    X: np.ndarray,
    k: int,
    metric: str,
    n_jobs: Optional[int],
    include_self: bool = True,
) -> sp.spmatrix:
    knn = kneighbors_graph(
        X,
        n_neighbors=k,
        mode="connectivity",
        metric=metric,
        include_self=include_self,
        n_jobs=n_jobs,
    )

    # RAC++ requires symmetric connectivity; sklearn also expects it for best results.
    knn = knn.maximum(knn.T)
    knn = knn.astype(bool)
    knn.sum_duplicates()
    knn.eliminate_zeros()
    return knn


def run_sklearn(
    X: np.ndarray,
    connectivity: sp.spmatrix,
    threshold: float,
    metric: str,
) -> Tuple[np.ndarray, float]:
    t0 = time.perf_counter()
    labels = AgglomerativeClustering(
        n_clusters=None,
        distance_threshold=threshold,
        linkage="average",
        metric=metric,
        connectivity=connectivity,
        compute_full_tree=True,
    ).fit_predict(X)
    return labels, time.perf_counter() - t0


def run_racplusplus(
    X: np.ndarray,
    connectivity: Optional[sp.spmatrix],
    threshold: float,
    batch_size: int,
    no_processors: int,
    metric: str,
) -> Tuple[np.ndarray, float]:
    import racplusplus

    t0 = time.perf_counter()
    labels = racplusplus.rac(X, threshold, connectivity, batch_size, no_processors, metric)
    return np.asarray(labels), time.perf_counter() - t0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compare sklearn AgglomerativeClustering vs RAC++ (kNN connectivity). Optionally also run RAC++ with full connectivity (no graph).",
    )
    parser.add_argument("--embeddings-npy", type=str, default=None, help="Path to a .npy embedding matrix (N, D).")
    parser.add_argument("--n", type=int, default=10_000, help="Number of rows to use from embeddings.")
    parser.add_argument("--dim", type=int, default=768, help="Dim used only when generating random embeddings.")
    parser.add_argument("--seed", type=int, default=42, help="RNG seed used only when generating random embeddings.")

    parser.add_argument("--k", type=int, default=30, help="k for kNN connectivity.")
    parser.add_argument("--metric", type=str, default="cosine", choices=["cosine", "euclidean"])
    parser.add_argument("--threshold", type=float, default=0.35, help="distance_threshold/max_merge_distance.")

    parser.add_argument("--batch-size", type=int, default=1000, help="RAC++ batch_size (used on connectivity path).")
    parser.add_argument("--no-processors", type=int, default=0, help="RAC++ thread count (0 = auto).")
    parser.add_argument("--n-jobs", type=int, default=None, help="Jobs for kNN graph construction (sklearn).")

    parser.add_argument("--skip-sklearn", action="store_true", help="Do not run sklearn.")
    parser.add_argument("--skip-rac", action="store_true", help="Do not run RAC++.")
    parser.add_argument(
        "--run-full-rac",
        action="store_true",
        help="Also run RAC++ with full connectivity (connectivity=None). WARNING: allocates an NxN distance matrix.",
    )
    args = parser.parse_args()

    if args.skip_sklearn and args.skip_rac and not args.run_full_rac:
        raise SystemExit("Nothing to run: both --skip-sklearn and --skip-rac were set (and --run-full-rac was not set).")

    if args.embeddings_npy:
        X = np.load(args.embeddings_npy)
        if X.ndim != 2:
            raise ValueError(f"Expected embeddings array with shape (N, D), got {X.shape}.")
        X = X[: args.n]
    else:
        rng = np.random.default_rng(args.seed)
        X = rng.random((args.n, args.dim), dtype=np.float64)

    X = np.asarray(X, dtype=np.float64, order="C")
    n = X.shape[0]
    print(f"Embeddings: shape={X.shape}, dtype={X.dtype}", flush=True)

    connectivity = None
    need_knn_connectivity = (not args.skip_sklearn) or (not args.skip_rac)
    if need_knn_connectivity:
        t0 = time.perf_counter()
        connectivity = build_knn_connectivity(X, k=args.k, metric=args.metric, n_jobs=args.n_jobs, include_self=True)
        knn_time = time.perf_counter() - t0

        nnz = int(connectivity.nnz)
        approx_density = nnz / float(n * n)
        print(
            f"kNN connectivity: k={args.k}, nnz={nnz}, density≈{approx_density:.3e}, build_time={knn_time:.3f}s",
            flush=True,
        )

    labels_sklearn = None
    labels_rac_knn = None
    labels_rac_full = None

    if not args.skip_sklearn:
        assert connectivity is not None
        labels_sklearn, t = run_sklearn(X, connectivity, threshold=args.threshold, metric=args.metric)
        print(f"sklearn: clusters={len(set(labels_sklearn))}, time={t:.3f}s", flush=True)

    if not args.skip_rac:
        assert connectivity is not None
        # pybind11 will accept CSR/CSC, but Eigen SparseMatrix defaults to column-major; CSC is typically fastest.
        labels_rac_knn, t = run_racplusplus(
            X,
            connectivity.tocsc(),
            threshold=args.threshold,
            batch_size=args.batch_size,
            no_processors=args.no_processors,
            metric=args.metric,
        )
        print(f"racplusplus (kNN): clusters={len(set(labels_rac_knn.tolist()))}, time={t:.3f}s", flush=True)

    if args.run_full_rac:
        est_bytes = int(n) * int(n) * 8  # Eigen::MatrixXd stores float64
        est_gib = est_bytes / float(1024**3)
        if est_gib >= 0.1:
            est_str = f"{est_gib:.2f} GiB"
        else:
            est_str = f"{est_bytes / float(1024**2):.1f} MiB"
        print(f"racplusplus (full): estimated distance matrix size ≈ {est_str}", flush=True)
        labels_rac_full, t = run_racplusplus(
            X,
            None,
            threshold=args.threshold,
            batch_size=args.batch_size,
            no_processors=args.no_processors,
            metric=args.metric,
        )
        print(f"racplusplus (full): clusters={len(set(labels_rac_full.tolist()))}, time={t:.3f}s", flush=True)

    if labels_sklearn is not None and labels_rac_knn is not None:
        ari = adjusted_rand_score(labels_sklearn, labels_rac_knn)
        nmi = normalized_mutual_info_score(labels_sklearn, labels_rac_knn)
        print(f"sklearn vs rac++ (kNN): ARI={ari:.4f}  NMI={nmi:.4f}", flush=True)

    if labels_rac_full is not None and labels_rac_knn is not None:
        ari = adjusted_rand_score(labels_rac_full, labels_rac_knn)
        nmi = normalized_mutual_info_score(labels_rac_full, labels_rac_knn)
        print(f"rac++ (full) vs rac++ (kNN): ARI={ari:.4f}  NMI={nmi:.4f}", flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
