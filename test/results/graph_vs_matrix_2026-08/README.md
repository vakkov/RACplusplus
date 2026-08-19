# Graph mode vs matrix mode — threshold exactness suite (2026-08-19)

Byte-level comparison of the matrix implementation (master code path) against
the matrix-free graph mode (`feature/graph-mode`, commit `b548fce`,
`RACPP_GRAPH_MODE=1`) across four cosine thresholds on the full real dataset.

## Setup

- **Data**: `~/agglomerative/auto-story-experiments/embeddings_100k.npy` —
  98,496 German news article embeddings, float32, 768-d, unit-normalized by
  the cosine path.
- **Machine**: AMD Ryzen 9 5950X (16C/32T), 62GB, THP=madvise, g++ 12.3,
  Eigen 3.4 (`eigen/` checkout). 24 threads, unbound.
- **Binary**: one build of `test/cli_npy.cpp` at commit `b548fce`
  (`-O3 -march=native -ffast-math -fopenmp`, `RACPP_SYMDIST_USE_FLOAT=1`,
  SIMD tail flags ON). The env var selects the mode at runtime; the matrix
  path is byte-for-byte the code verified against master in
  `test/perf_verification_2026-08.md`.
- **Command**: `[RACPP_GRAPH_MODE=1] racpp_ab embeddings_100k.npy <T> 24`
- Labels (one per line, 98,496 lines) in this directory as
  `labels_98k_t<T>_{matrix,graph}.txt`; full stderr in `log_t<T>_*.log`.

## Results — labels byte-identical at every threshold

| T | clusters | labels | matrix wall / peakRSS | graph wall / peakRSS | edges | avg deg | graph iters / nn_recompute |
|-----|--------|-----------|------------------|-----------------|-------|-------|------------------|
| 0.2 | 55,716 | IDENTICAL | 27.6s / 19.37GB | **9.5s / 0.99GB** | 1.41M | 28.7 | 101 / 0.40s |
| 0.3 | 45,304 | IDENTICAL | 28.8s / 19.38GB | **9.8s / 1.00GB** | 1.90M | 38.5 | 115 / 0.58s |
| 0.4 | 30,636 | IDENTICAL | 30.7s / 19.38GB | **11.6s / 1.02GB** | 3.94M | 80.0 | 133 / 1.67s |
| 0.5 | 15,420 | IDENTICAL | 32.5s / 19.38GB | **25.1s / 1.14GB** | 16.3M | 331.4 | 147 / 10.48s |

MD5 checksums (matrix and graph files are identical pairs):

```
a09afb3cf62cd38d9d170c648c9b2d60  labels_98k_t0.2_{matrix,graph}.txt
5f05dcda9969099a7c9d6b86e20b85d1  labels_98k_t0.3_{matrix,graph}.txt
6be887a521d17e16f5f09d18c433bcd1  labels_98k_t0.4_{matrix,graph}.txt
90b198a4eed02e59cfffeba053e2c9b1  labels_98k_t0.5_{matrix,graph}.txt
```

## Reading the numbers

- **Exactness**: identical output at 4 thresholds spanning 55.7k -> 15.4k
  clusters (light to heavy merging). Together with earlier runs this makes 9
  verified configurations (incl. T=0.35 full-98k, 30k slices, synthetic
  double path) with zero label differences.
- **The density model in action**: graph-mode cost tracks edge density
  exactly as predicted. At T=0.2-0.4 (avg degree 29-80) the merge loop is
  0.4-1.7s and graph mode is ~3x faster overall. At T=0.5 (avg degree 331,
  16.3M edges) nn_recompute grows to 10.5s and the advantage narrows to
  1.3x - the approach to the crossover regime. Memory stays ~17x smaller
  even there.
- **Guidance**: for production story-clustering thresholds (~0.3-0.4 cosine)
  graph mode is comfortably in its best regime. Beyond ~0.5 on this data,
  measure first (`Graph init` prints edges/avg_deg immediately; abort if
  avg_deg is in the thousands) or stay on matrix mode.

## Reproduce

```bash
git checkout feature/graph-mode
g++ -O3 -std=c++17 -march=native -ffast-math -fopenmp \
    -I eigen -I src/racplusplus \
    -DRACPP_BUILDING_LIB_ONLY=1 -DRACPP_SYMDIST_USE_FLOAT=1 \
    test/cli_npy.cpp -o racpp_ab
E=~/agglomerative/auto-story-experiments/embeddings_100k.npy
for t in 0.2 0.3 0.4 0.5; do
  ./racpp_ab $E $t 24 > m.txt
  RACPP_GRAPH_MODE=1 ./racpp_ab $E $t 24 > g.txt
  diff -q m.txt g.txt && echo "t=$t identical"
done
```
