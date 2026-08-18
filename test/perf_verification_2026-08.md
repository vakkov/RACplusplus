# Performance optimization verification — August 2026

Record of the output-preservation and performance verification for the
`SymDistBuffer` change (committed as `c469033`) and the dense-init tile
tuning. All comparisons are **baseline = unmodified source at commit
`7a908dc`** vs **optimized = same source + the SymDistBuffer change**, both
compiled fresh with identical flags. No prebuilt wheels or old CLI binaries
were used anywhere.

## Environment

- AMD Ryzen 9 5950X (16C/32T), 62GB RAM, Linux 6.8, THP=madvise
- g++ 12.3.0, Eigen 3.4 (`eigen/` checkout)
- Flags: `-O3 -std=c++17 -march=native -ffast-math -fopenmp`,
  `RACPP_SYMDIST_USE_FLOAT=1`, both SIMD tail flags ON
- Driver: `test/cli_npy.cpp` (real data, exercises the pybind float32 path)
  and a synthetic-data driver calling `RAC()` (double path)

## What the patch does

Replaces `std::vector<SymDistScalar>` for `SymDistMatrix::data` with an
mmap-backed move-only buffer (malloc fallback on non-Linux):

1. no value-initialization — removes the single-threaded infinity fill
   (~280ms/20k pts, ~7s/98k pts measured at 2.9GB/s) and moves NUMA
   first-touch into the parallel init tile loop;
2. `madvise(MADV_HUGEPAGE)` — hugepages without env tunables (unit test:
   762MB of a 766MB buffer adopted);
3. shrink releases tail pages — RSS drops at each compaction instead of
   holding the init peak for the whole run.

Safety invariant: `dist.data` is uninitialized until dense init writes every
element. Proven by (a) poison-fill equivalence — filling with 0.0 instead of
infinity produces bit-identical labels, so no fill value is ever observable —
and (b) valgrind memcheck on a no-init build: 0 errors (N=1200, both metrics).

## Output preservation — all label vectors bit-identical

Synthetic (20,000 x 768, double input path, full label-vector diffs):

| config | result |
|---|---|
| cosine thr .010 / .020 (no merges, 0 compactions) | identical |
| cosine thr .035 (1966 clusters, 2 compactions) | identical |
| cosine thr .040 / .045 / .050 / .070 (heavy merge) | identical |
| cosine .035, 1 thread | identical |
| cosine .035, N=2001 (partial tiles) | identical |
| euclidean 5.6 (1977 clusters) | identical |
| TILE=192 vs 768 | identical |
| run-to-run determinism | identical |

Real data (`embeddings_100k.npy`: 98,496 German news articles, 768-d float32,
threshold **0.35** cosine — the production story-clustering regime, 8 threads):

| config | result |
|---|---|
| 30,000-row slice | identical (13,603 clusters) |
| full 98,496 | **identical (38,443 clusters)** |
| full 98,496, determinism re-run | identical |

## Performance — full 98,496 articles, 0.35 cosine, 8 threads

| metric | baseline | patch + `RACPP_DENSE_INIT_TILE=192` |
|---|---|---|
| total wall | 43.3s | **34.3s (-21%)** |
| init phase | 21.1s | 13.6s (-36%) |
| RAC_i (dissim+nn) | ~20.5s | ~20.1s (unchanged code) |
| RSS profile | flat ~19.3GB entire run | 19.3GB peak, **drops to 6.5GB** at the compaction (98496->57142, ~28s in) |

Run shape at 0.35 on real data: 127 iterations, one compaction, 38,443 final
clusters. Peak memory is unchanged (the matrix must exist through the early
iterations); the gain is the post-compaction release plus init time.

Separately measured on the synthetic 20k benchmark: dense-init tile default
768 is suboptimal on Zen 3 — TILE=192 gives init 1175ms -> ~815ms (sweep:
192=802, 256=886, 512=1012, 768=1238, 1536=1546ms). Sweep again per machine
and per dtype via the `RACPP_DENSE_INIT_TILE` env var.

## Reproducing

```bash
# build both sides (baseline: checkout 7a908dc; optimized: c469033 or later)
g++ -O3 -std=c++17 -march=native -ffast-math -fopenmp \
    -I eigen -I src/racplusplus \
    -DRACPP_BUILDING_LIB_ONLY=1 -DRACPP_SYMDIST_USE_FLOAT=1 \
    -DRACPP_SIMD_DISSIM_TAIL_UPDATE=1 -DRACPP_SIMD_NN_TAIL_UPDATE=1 \
    test/cli_npy.cpp -o racpp_npy
# run + diff (labels on stdout, diagnostics on stderr)
./racpp_npy_baseline embeddings_100k.npy 0.35 8 > base.txt
RACPP_DENSE_INIT_TILE=192 ./racpp_npy_patched embeddings_100k.npy 0.35 8 > opt.txt
diff -q base.txt opt.txt
```

## Deployment notes (prod = latest master, compiled on the EPYC 7513 server)

- Because prod builds from master on the server itself, these results apply
  directly: the patch changes allocation only (no arithmetic, no iteration
  order), so its output-identity is compiler-independent. `-march=native` on
  the EPYC resolves to znver3 — correct AVX2/FMA codegen; the ISA-mismatch
  risk only applies to wheels built elsewhere (e.g. CI).
- The one build-specific empirical claim is TILE=192 bit-identity (a property
  of the Eigen GEMM blocking under the local compiler). Re-check it on the
  prod gcc/Eigen with the diff above — ~40s per side at 98k.
- The dev-box wheel with a 7th `rac()` argument ("generic") is a stale local
  experiment; ignore it.
- Prod is a **dual-socket 2x EPYC 7513** (numactl: 2 nodes, 256GB each,
  distance 10/32). Run the job bound to one socket:
  `numactl --cpunodebind=0 --preferred=0 ...` with no_processors=16-32.
  One socket (32C, 8 DDR4 channels ~205GB/s, 256GB) is ample; unbound runs
  let the serial fill park the 19GB matrix on one node while threads migrate
  across sockets paying ~3.2x remote latency. Binding helps the CURRENT code
  immediately, independent of the patch. `--interleave=0,1` only if ever
  running >64 threads across both sockets. (`numactl -H` "free" excludes
  reclaimable cache — check real headroom with `free -g`/`numastat -m`.)
- Remaining one-time sweeps on prod: thread count (16/32 bound to a socket),
  `RACPP_DENSE_INIT_TILE` (192/256/384; expect more init gain than on the
  desktop — the removed serial fill and hugepage/TLB effects grow with the
  19GB+ matrix, and lower server clocks make serial sections costlier).
