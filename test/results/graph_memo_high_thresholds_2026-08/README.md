# Graph mode: cached-value store at high thresholds (2026-08-19)

Experiments for `feature/graph-mode-lw-store` — memoizing cluster distances in
the candidate lists to cut the per-look cost that dominates graph mode when the
merge threshold (and hence candidate density) is high.

## What changed

Candidate lists went from `vector<int>` to `vector<Cand{id, ver, mean}>`
(16 bytes/entry). A `version[c]` counter increments whenever cluster `c`
absorbs another cluster. During a rescan of cluster X, an entry's cached mean
is reused **only** when X itself did not change this round and
`version[id]` still matches the value stamped on the entry; otherwise the mean
is recomputed with the same double closed form as before. Reused values are
therefore bit-identical to recomputed ones — this is a pure memoization, not a
new formula.

**Why not literal Lance-Williams.** Storing pairwise distance *sums* and
combining them additively on merge (the classic LW update) is unsound here:
the sum of a merged pair needs the point pairs that were never discovered
(those >= T at init), so it would systematically underestimate and cause
spurious merges. Mixing LW-derived values with closed-form values for the same
pair also reintroduces endpoint disagreement, which is what deadlocked the
first graph-mode prototype on duplicate cliques. Memoization keeps exactly one
value system.

## Exactness

All runs below produced labels **byte-identical to both** the matrix mode and
the pre-memo graph mode (`b548fce`), including the extreme T=0.6 case:

| test | result |
|---|---|
| 98k @ T=0.2 / 0.3 / 0.4 / 0.5 | identical to recorded matrix + graph labels |
| 98k @ T=0.6 (84M edges) | identical to matrix (5,500 clusters) |
| 30k @ T=0.35 / 0.5 / 0.6 | identical to matrix and pre-memo graph |
| determinism (98k @ T=0.5, repeat run) | identical |
| thread invariance (8 vs 32 threads) | identical |

## Performance — full 98,496 articles, 24 threads, Ryzen 5950X

`nn_recompute` is the merge-loop cost the memo targets; wall includes the
shared init GEMM.

| T | edges | avg deg | nn_recompute before | after | loop speedup | wall (memo) | wall (matrix) | RSS (memo) | RSS (matrix) |
|-----|-------|--------|--------------|--------|------|--------|--------|--------|---------|
| 0.2 | 1.4M | 29 | 0.40s | **0.18s** | 2.2x | 9.2s | 27.6s | ~1.0GB | 19.4GB |
| 0.3 | 1.9M | 38 | 0.58s | **0.25s** | 2.3x | 9.4s | 28.8s | ~1.0GB | 19.4GB |
| 0.4 | 3.9M | 80 | 1.67s | **0.73s** | 2.3x | 10.5s | 30.7s | 1.02GB | 19.4GB |
| 0.5 | 16.3M | 331 | 10.48s | **4.41s** | 2.4x | 19.5s | 32.5s | 1.32GB | 19.4GB |
| 0.6 | 84.2M | 1709 | — | 26.55s | — | 73.8s | 34.5s | 3.74GB | 19.4GB |

Cache hit rate falls with density (63% at T=0.2, 55% at T=0.4, 50% at T=0.5,
49% at T=0.6): misses are dominated by merged mains, whose entire candidate
list is invalidated by their own change. Bystander rescans are where the
memo pays.

## Where the crossover is

- **Time**: graph mode (with memo) wins up to T~0.5 on this data at 98k; at
  T=0.6 the matrix wins (34.5s vs 73.8s). At 30k the crossover is earlier
  (T=0.6: matrix 3.4s vs graph 5.5s) because a 30k matrix is only 1.9GB.
- **Memory**: graph mode wins everywhere — 5x less even in the T=0.6 worst
  case, 19x less in the production range.
- **Init dominates at extreme T**: at T=0.6 edge collection alone is 45s of
  the 73.8s, i.e. the cost of materializing 84M edges, not of the merge loop.

Practical reading: for production story-clustering thresholds (0.3-0.4,
avg degree 29-80) the memo makes the merge loop essentially free (<1s) and
graph mode is ~3x faster than matrix at 19x less memory. Past avg degree
~1000, use matrix mode if it fits in RAM, graph mode if it does not.

## Production validation (2026-08-20, dual EPYC 7513, 32 threads socket-bound)

Run on PRD-01-GPU-01 with `numactl --cpunodebind=0 --preferred=0`, same 98k
embedding file. Exactness: **all three implementations byte-identical at
T=0.35 / 0.5 / 0.6** (38,443 / 15,420 / 5,500 clusters — matching the dev
results exactly).

| T | matrix | plain graph | memo graph | best |
|-----|--------------|--------------|--------------|------|
| 0.35 | 12.8-13.2s / 19.4GB | 6.1-6.4s / 1.03GB | **5.8s / 1.04GB** | memo |
| 0.5 | 14.0-14.4s / 19.4GB | 11.8s / 1.15GB | **9.9-10.1s / 1.35GB** | memo |
| 0.6 | **14.3-14.5s / 19.4GB** | 49.3-49.7s / 1.8GB | 32.4-32.8s / 3.7GB | matrix |

The EPYC's 8-channel bandwidth flatters matrix mode at high T (its 0.6 run is
2.4x faster than on the desktop), so the time crossover lands at ~T=0.5 there
vs ~T=0.55-0.6 on the desktop. The memo beat plain graph at every threshold
on both machines.

**Service routing (adopted):** `RACPP_GRAPH_MODE=1` for jobs with T <= 0.5;
unset (matrix) for T >= 0.6 while N x N/2 x 4B fits node RAM. Both routes are
output-identical, so routing is purely operational.

## Reproduce

```bash
git checkout feature/graph-mode-lw-store
g++ -O3 -std=c++17 -march=native -ffast-math -fopenmp \
    -I eigen -I src/racplusplus \
    -DRACPP_BUILDING_LIB_ONLY=1 -DRACPP_SYMDIST_USE_FLOAT=1 \
    test/cli_npy.cpp -o racpp_memo
E=~/agglomerative/auto-story-experiments/embeddings_100k.npy
for t in 0.2 0.3 0.4 0.5 0.6; do
  RACPP_GRAPH_MODE=1 /usr/bin/time -f "T=$t memo %es %MkB" ./racpp_memo $E $t 24 > g.txt
  ./racpp_memo $E $t 24 > m.txt            # matrix mode, same binary
  diff -q g.txt m.txt && echo "T=$t identical"
done
```
