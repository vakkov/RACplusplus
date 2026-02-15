from ._racplusplus import (
    rac as _rac_native,
    test_rac,
    simple_pybind_io_test,
    _pairwise_euclidean_distance,
    _pairwise_cosine_distance,
    __symdist_storage__,
)


def rac(
    base_arr,
    max_merge_distance,
    connectivity=None,
    batch_size=0,
    no_processors=0,
    distance_metric="euclidean",
):
    """
    Run Reciprocal Agglomerative Clustering.

    Parameters:
    - base_arr: numpy.ndarray with shape (N, D), float64, C-contiguous
    - max_merge_distance: float
    - connectivity: optional scipy sparse matrix with shape (N, N)
    - batch_size: int, default 0 (auto)
    - no_processors: int, default 0 (auto)
    - distance_metric: "euclidean" or "cosine"
    """
    return _rac_native(
        base_arr,
        max_merge_distance,
        connectivity,
        batch_size,
        no_processors,
        distance_metric,
    )


def rac_help():
    """Return the RAC call signature and parameter summary."""
    return (
        "rac(base_arr, max_merge_distance, connectivity=None, "
        "batch_size=0, no_processors=0, distance_metric='euclidean')\n"
        "base_arr: float64 C-contiguous ndarray of shape (N, D)\n"
        "connectivity: optional sparse (N, N)\n"
        "distance_metric: 'euclidean' or 'cosine'"
    )


__all__ = (
    "rac",
    "rac_help",
    "test_rac",
    "simple_pybind_io_test",
    "_pairwise_euclidean_distance",
    "_pairwise_cosine_distance",
    "__symdist_storage__",
)
