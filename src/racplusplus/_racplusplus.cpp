#ifndef RACPP_BUILDING_LIB_ONLY
#define RACPP_BUILDING_LIB_ONLY 0
#endif

#if !RACPP_BUILDING_LIB_ONLY
#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
namespace py = pybind11;
#endif

#include <array>
#include <tuple>
#include <set>
#include <chrono>
#include <vector>
#include <limits>
#include <iostream>
#include <thread>
#include <algorithm>
#include <memory>
#include <functional>
#include <mutex>
#include <condition_variable>
#include <type_traits>
#include <atomic>
// #define EIGEN_DONT_PARALLELIZE
#include "Eigen/Dense"
#include "Eigen/Sparse"
#include <random>
#include <numeric>
#include <cmath>
#include <cstdlib>
#if defined(RACPP_SIMD_TAIL_UPDATE) && RACPP_SIMD_TAIL_UPDATE
#include <immintrin.h>
#endif

#include "_racplusplus.h"

// Timing/stat tracking globals
std::vector<long> UPDATE_NEIGHBOR_DURATIONS;
std::vector<long> UPDATE_NN_DURATIONS;
std::vector<long> COSINE_DURATIONS;
std::vector<long> INDICES_DURATIONS;
std::vector<long> MERGE_DURATIONS;
std::vector<long> MISC_MERGE_DURATIONS;
std::vector<long> INITIAL_NEIGHBOR_DURATIONS;
std::vector<long> HASH_DURATIONS;
std::vector<double> UPDATE_PERCENTAGES;

struct DenseOpsProfile {
    long long dissim_cross_ns = 0;
    long long dissim_candidate_build_ns = 0;
    long long dissim_compute_ns = 0;
    long long dissim_writeback_ns = 0;
    long long dissim_last_batch_nn_ns = 0;
    long long nn_changed_shortlist_prep_ns = 0;
    long long nn_scan_run_build_ns = 0;
    long long nn_rescan_total_ns = 0;
    long long nn_shortlist_eval_ns = 0;
    long long nn_fullscan_eval_ns = 0;
    uint64_t nn_rescan_clusters = 0;
    uint64_t nn_shortlist_attempts = 0;
    uint64_t nn_shortlist_hits = 0;
    uint64_t nn_fullscan_clusters = 0;

    void reset() { *this = DenseOpsProfile{}; }
};

static DenseOpsProfile g_dense_ops_profile;

static inline bool racpp_profile_ops_enabled() {
    static int cached = -1;
    if (cached < 0) {
        const char* env = std::getenv("RACPP_PROFILE_OPS");
        cached = (env != nullptr && env[0] != '\0' && env[0] != '0') ? 1 : 0;
    }
    return cached == 1;
}

static inline long long ns_between(
    const std::chrono::high_resolution_clock::time_point& t0,
    const std::chrono::high_resolution_clock::time_point& t1) {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();
}

static inline double ns_to_ms(long long ns) {
    return static_cast<double>(ns) / 1e6;
}

#if defined(RACPP_SIMD_TAIL_UPDATE) && RACPP_SIMD_TAIL_UPDATE && \
    defined(__AVX2__) && defined(__FMA__) && \
    defined(RACPP_SYMDIST_USE_FLOAT) && RACPP_SYMDIST_USE_FLOAT
static inline float hmin_ps256(__m256 v) {
    __m128 lo = _mm256_castps256_ps128(v);
    __m128 hi = _mm256_extractf128_ps(v, 1);
    __m128 m = _mm_min_ps(lo, hi);
    m = _mm_min_ps(m, _mm_movehl_ps(m, m));
    m = _mm_min_ps(m, _mm_shuffle_ps(m, m, 0x55));
    return _mm_cvtss_f32(m);
}
#endif

//get number of processors
size_t getProcessorCount() {
    const auto NO_PROCESSORS = std::thread::hardware_concurrency();
    std::cout << "Processors: " << NO_PROCESSORS << std::endl;
    return NO_PROCESSORS != 0 ? static_cast<size_t>(NO_PROCESSORS) : static_cast<size_t>(8);
}

std::string vectorToString(const std::vector<std::pair<int, int>>& merges) {
    std::ostringstream oss;
    oss << "[";
    for (auto it = merges.begin(); it != merges.end(); ++it) {
        oss << "(" << it->first << ", " << it->second << ")";
        if (std::next(it) != merges.end()) {
            oss << ", ";
        }
    }
    oss << "]";
    return oss.str();
}

class TinyThreadPool {
public:
    explicit TinyThreadPool(size_t thread_count)
        : thread_count_(std::max<size_t>(1, thread_count)) {
        if (thread_count_ <= 1) {
            return;
        }
        workers_.reserve(thread_count_ - 1);
        for (size_t worker_id = 1; worker_id < thread_count_; ++worker_id) {
            workers_.emplace_back([this, worker_id]() { worker_loop(worker_id); });
        }
    }

    ~TinyThreadPool() {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            stopping_ = true;
            ++generation_;
        }
        cv_.notify_all();
        for (auto& worker : workers_) {
            worker.join();
        }
    }

    template <typename Fn>
    void parallel_for(size_t total, Fn&& fn) {
        if (total == 0) {
            return;
        }
        if (thread_count_ <= 1 || total == 1) {
            fn(0, total);
            return;
        }

        const size_t active_threads = std::min(thread_count_, total);
        if (active_threads <= 1) {
            fn(0, total);
            return;
        }

        {
            std::lock_guard<std::mutex> lock(mutex_);
            active_threads_ = active_threads;
            total_work_ = total;
            remaining_workers_ = active_threads_ - 1;
            job_fn_ = [&fn](size_t start, size_t end) { fn(start, end); };
            ++generation_;
        }
        cv_.notify_all();

        auto [main_start, main_end] = chunk_for_worker(0, total, active_threads);
        if (main_start < main_end) {
            job_fn_(main_start, main_end);
        }

        std::unique_lock<std::mutex> lock(mutex_);
        done_cv_.wait(lock, [this]() { return remaining_workers_ == 0; });
    }

private:
    using JobFn = std::function<void(size_t, size_t)>;

    static std::pair<size_t, size_t> chunk_for_worker(
        size_t worker_id,
        size_t total,
        size_t active_threads) {
        const size_t chunk = total / active_threads;
        const size_t remainder = total % active_threads;
        const size_t start = worker_id * chunk + std::min(worker_id, remainder);
        const size_t end = start + chunk + (worker_id < remainder ? 1 : 0);
        return {start, end};
    }

    void worker_loop(size_t worker_id) {
        size_t seen_generation = 0;
        while (true) {
            JobFn fn;
            size_t total = 0;
            size_t active = 0;
            {
                std::unique_lock<std::mutex> lock(mutex_);
                cv_.wait(lock, [this, seen_generation]() {
                    return stopping_ || generation_ != seen_generation;
                });

                if (stopping_) {
                    return;
                }

                seen_generation = generation_;
                total = total_work_;
                active = active_threads_;
                if (worker_id >= active) {
                    continue;
                }
                fn = job_fn_;
            }

            auto [start, end] = chunk_for_worker(worker_id, total, active);
            if (start < end) {
                fn(start, end);
            }

            std::lock_guard<std::mutex> lock(mutex_);
            if (remaining_workers_ > 0) {
                --remaining_workers_;
                if (remaining_workers_ == 0) {
                    done_cv_.notify_one();
                }
            }
        }
    }

    size_t thread_count_;
    std::vector<std::thread> workers_;

    std::mutex mutex_;
    std::condition_variable cv_;
    std::condition_variable done_cv_;

    bool stopping_ = false;
    size_t generation_ = 0;
    size_t active_threads_ = 1;
    size_t total_work_ = 0;
    size_t remaining_workers_ = 0;
    JobFn job_fn_;
};

static TinyThreadPool& get_thread_pool(size_t requested_threads) {
    const size_t threads = std::max<size_t>(1, requested_threads);
    thread_local std::unique_ptr<TinyThreadPool> pool;
    thread_local size_t pool_threads = 0;
    if (!pool || pool_threads != threads) {
        pool = std::make_unique<TinyThreadPool>(threads);
        pool_threads = threads;
    }
    return *pool;
}

template <typename Fn>
static inline void run_parallel_for(size_t requested_threads, size_t total, Fn&& fn) {
    if (total == 0) {
        return;
    }
    const size_t threads = std::max<size_t>(1, requested_threads);
    if (threads <= 1 || total <= 1) {
        fn(0, total);
        return;
    }
    TinyThreadPool& pool = get_thread_pool(threads);
    pool.parallel_for(total, std::forward<Fn>(fn));
}

static inline int resolve_processor_count(int no_processors) {
    if (no_processors > 0) {
        return no_processors;
    }
    const unsigned int hc = std::thread::hardware_concurrency();
    return (hc != 0) ? static_cast<int>(hc) : 8;
}

//--------------------DSU (Disjoint Set Union)------------------------------------
static inline int dsu_find(std::vector<int>& parent, int x) {
    while (parent[x] != x) {
        parent[x] = parent[parent[x]]; // path halving
        x = parent[x];
    }
    return x;
}

static inline void dsu_union(std::vector<int>& parent, std::vector<int>& size, int main, int secondary) {
    // main always becomes the root (preserves cluster id convention)
    parent[secondary] = main;
    size[main] += size[secondary];
}
//--------------------End DSU------------------------------------

//----standalone test driver
int racplusplus_cli_test() {
    std::cout << std::endl;
    std::cout << "Starting Randomized RAC Test" << std::endl;
    std::cout << "Number of Processors Found for Program Use: " << getProcessorCount() << std::endl;
    // 5000 - 1061
    const int NO_POINTS = 20000;
    Eigen::MatrixXd test = generateRandomMatrix(NO_POINTS, 768, 10);
    // Shift and scale the values to the range 0-1
    test = (test + Eigen::MatrixXd::Constant(NO_POINTS, 768, 1.)) / 2.;

    Eigen::SparseMatrix<bool> connectivity;
    //set up test
    double max_merge_distance = .035;
    int batch_size = 100;
    int no_processors = 8;
    //actually run test
    std::vector<int> labels = RAC(test, max_merge_distance, nullptr, batch_size, no_processors, "cosine");

    // Output duration
    std::cout << std::accumulate(UPDATE_NEIGHBOR_DURATIONS.begin(), UPDATE_NEIGHBOR_DURATIONS.end(), 0.0) / 1000 << std::endl;

    // Output NN update durations
    std::cout << std::accumulate(UPDATE_NN_DURATIONS.begin(), UPDATE_NN_DURATIONS.end(), 0.0) / 1000 << std::endl;

    // Output indices durations
    std::cout << std::accumulate(INDICES_DURATIONS.begin(), INDICES_DURATIONS.end(), 0.0) / 1000 << std::endl;

    // Output merge durations
    std::cout << std::accumulate(MERGE_DURATIONS.begin(), MERGE_DURATIONS.end(), 0.0) / 1000 << std::endl;

    // Output misc merge durations
    std::cout << std::accumulate(MISC_MERGE_DURATIONS.begin(), MISC_MERGE_DURATIONS.end(), 0.0) / 1000 << std::endl;

    // Output number of clusters
    std::set<int> unique_labels(labels.begin(), labels.end());
    std::cout << "Unique labels: " << unique_labels.size() << std::endl;

    std::cout << std::endl;
    return 0;
}

//---------------------Classes------------------------------------


Cluster::Cluster(int id)
    : id(id), will_merge(false), active(true) {
        indices.push_back(id);
        this->nn = -1;
        this->nn_distance = std::numeric_limits<double>::infinity();
    }


void Cluster::update_nn(double max_merge_distance) {
    if (neighbor_distances.size() == 0) {
        nn = -1;
        nn_distance = std::numeric_limits<double>::infinity();
        return;
    }

    double min = std::numeric_limits<double>::infinity();
    int nn = -1;

    for (auto& neighbor : this->neighbor_distances) {
        double dissimilarity = neighbor.second;
        if (dissimilarity < min) {
            min = dissimilarity;
            nn = neighbor.first;
        }
    }

    nn_distance = min;
    if (min < max_merge_distance) {
        this->nn = nn;
    } else {
        this->nn = -1;
    }
}

void Cluster::update_nn(const SymDistMatrix& dist, double max_merge_distance) {
    auto [min_val, min_idx] = dist.min_in_col(this->id);
    nn_distance = min_val;

    if (min_val < max_merge_distance) {
        this->nn = min_idx;
    } else {
        this->nn = -1;
    }
}

//---------------------End Classes------------------------------------


//--------------------Helpers------------------------------------

void printMatrixInfo(Eigen::MatrixXd& matrix) {
    // Count the number of infinity, negative, and over 0.3 elements
    int infCount = (matrix.array() == std::numeric_limits<double>::infinity()).count();
    int negCount = (matrix.array() < 0.0).count();
    int overCount = (matrix.array() > 0.5).count();

    std::cout << "Number of inf elements: " << infCount << std::endl;
    std::cout << "Number of negative elements: " << negCount << std::endl;
    std::cout << "Number of elements over 0.5: " << overCount << std::endl;
}

// Function to generate a matrix filled with random numbers.
Eigen::MatrixXd generateRandomMatrix(int rows, int cols, int seed) {
    std::default_random_engine generator(seed);
    std::uniform_real_distribution<double> distribution(0.0,1.0);

    Eigen::MatrixXd mat(rows, cols);

    int numRows = mat.rows();
    int numCols = mat.cols();
    for(int i=0; i<numRows; ++i) {
        for(int j=0; j<numCols; ++j) {
            mat(i, j) = distribution(generator);
        }
    }

    return mat;
}

double get_arr_value(Eigen::MatrixXd& arr, int i, int j) {
    if (i > j) {
        return arr(j, i);
    }
    return arr(i, j);
}

void set_arr_value(Eigen::MatrixXd& arr, int i, int j, double value) {
    if (i > j) {
        arr(j, i) = value;
        return;
    }
    arr(i, j) = value;
}

void remove_secondary_clusters(
    std::vector<std::pair<int, int> >& merges,
    std::vector<Cluster>& clusters,
    std::vector<int>& active_indices,
    std::vector<int>& active_pos) {
    for (const auto& merge : merges) {
        const int secondary = merge.second;
        if (!clusters[secondary].active) {
            continue;
        }

        clusters[secondary].active = false;

        const int pos = active_pos[secondary];
        if (pos < 0) {
            continue;
        }

        const int last_idx = static_cast<int>(active_indices.size()) - 1;
        const int moved_id = active_indices[last_idx];
        if (pos != last_idx) {
            active_indices[pos] = moved_id;
            active_pos[moved_id] = pos;
        }

        active_indices.pop_back();
        active_pos[secondary] = -1;
    }
}
//--------------------End Helpers------------------------------------

//-----------------------Distance Calculations-------------------------

Eigen::MatrixXd pairwise_cosine(const Eigen::MatrixXd& A, const Eigen::MatrixXd& B) {
    Eigen::MatrixXd D = A.transpose() * B;   // one N×N buffer
    D = (-D).array() + 1.0;                  // reuse the same buffer
    return D;
}


//Calculate pairwise euclidean between two matrices
Eigen::MatrixXd pairwise_euclidean(const Eigen::MatrixXd& array_a, const Eigen::MatrixXd& array_b) {
    Eigen::MatrixXd D = -2.0 * array_a.transpose() * array_b;
    D.colwise() += array_a.colwise().squaredNorm().transpose();
    D.rowwise() += array_b.colwise().squaredNorm();
    return D.array().sqrt();
}

// //Averaged dissimilarity across two matrices (wrapper for pairwise distance calc + avging)
// // Fix #4: take by const reference instead of by value
// double calculate_weighted_dissimilarity(const Eigen::MatrixXd& points_a, const Eigen::MatrixXd& points_b) {
//     Eigen::MatrixXd dissimilarity_matrix = pairwise_cosine(points_a, points_b);

//     return static_cast<double>(dissimilarity_matrix.mean());
// }

// Fix #4: take clusters by const reference instead of by value
std::vector<std::pair<int, std::vector<std::pair<int, double>>>> consolidate_indices(
    std::vector<int>& sort_neighbor_arr,
    std::vector<std::pair<int, int> >& merges,
    std::vector<Cluster>& clusters) {

    std::vector<std::pair<int, std::vector<std::pair<int, double>>>> return_vectors;

    int vector_idx = 0;
    for (const auto& merge : merges) {
        int main = merge.first;
        int secondary = merge.second;

        clusters[main].indices.insert(clusters[main].indices.end(), clusters[secondary].indices.begin(), clusters[secondary].indices.end());
        clusters[secondary].indices.clear();

        for (size_t i=0; i < clusters[main].neighbors_needing_updates.size(); i++) {
            int neighbor_idx = std::get<1>(clusters[main].neighbors_needing_updates[i]);
            double dissimilarity = std::get<2>(clusters[main].neighbors_needing_updates[i]);

            if (sort_neighbor_arr[neighbor_idx] == -1) {
                sort_neighbor_arr[neighbor_idx] = vector_idx;
                return_vectors.push_back(std::make_pair(neighbor_idx, std::vector<std::pair<int, double>>()));
                vector_idx++;
            }

            return_vectors[sort_neighbor_arr[neighbor_idx]].second.push_back(std::make_pair(main, dissimilarity));
        }
    }

    return return_vectors;
}

void update_cluster_dissimilarities(
    std::vector<std::pair<int, int> >& merges,
    std::vector<Cluster>& clusters,
    const int NO_PROCESSORS,
    Eigen::MatrixXd& base_arr) {

    static std::vector<std::vector<int>> merging_arrays(NO_PROCESSORS, std::vector<int>(clusters.size()));

    if (merges.size() / NO_PROCESSORS > 10) {
        parallel_merge_clusters(merges, clusters, NO_PROCESSORS, merging_arrays, base_arr);
    } else {
        for (auto& merge : merges) {
            merge_cluster_compute_linkage(merge, clusters, merging_arrays[0], base_arr);
        }
    }

    static std::vector<int> sort_neighbor_arr(clusters.size(), -1);
    std::vector<std::pair<int, std::vector<std::pair<int, double>>>> neighbor_updates = consolidate_indices(sort_neighbor_arr, merges, clusters);

    static std::vector<int> update_neighbors_arr(clusters.size());
    for (size_t i=0; i<neighbor_updates.size(); i++) {
        update_cluster_neighbors(neighbor_updates[i], clusters, update_neighbors_arr);
        sort_neighbor_arr[neighbor_updates[i].first] = -1;
    }
}

void update_cluster_dissimilarities(
    std::vector<std::pair<int, int> >& merges,
    std::vector<Cluster>& clusters,
    const int NO_PROCESSORS,
    std::vector<std::vector<std::pair<int, double>>>& merging_arrays,
    std::vector<int>& sort_neighbor_arr,
    std::vector<std::vector<int>>& update_neighbors_arrays
    ) {

    if (merges.size() / NO_PROCESSORS > 10) {
        parallel_merge_clusters(merges, clusters, NO_PROCESSORS, merging_arrays);
    } else {
        for (auto& merge : merges) {
            merge_cluster_symmetric_linkage(merge, clusters, merging_arrays[0]);
        }
    }

    std::vector<std::pair<int, std::vector<std::pair<int, double>>>> neighbor_updates = consolidate_indices(sort_neighbor_arr, merges, clusters);
    parallel_update_clusters(
        neighbor_updates,
        clusters,
        update_neighbors_arrays,
        sort_neighbor_arr,
        NO_PROCESSORS);
}

static inline size_t choose_merge_batch_size(int n, size_t requested_threads) {
    // Target workspace budget for merged columns:
    //   batch_size * N * sizeof(SymDistScalar)
    // Keep this bounded to reduce memory pressure/cache thrash at large N.
    constexpr size_t TARGET_WORKSPACE_BYTES = 256ull * 1024ull * 1024ull; // 256 MB
    constexpr size_t MIN_BATCH = 128;
    constexpr size_t MAX_BATCH = 1024;

    const size_t n_safe = std::max<size_t>(1, static_cast<size_t>(n));
    const size_t bytes_per_col = n_safe * sizeof(SymDistScalar);
    size_t by_memory = TARGET_WORKSPACE_BYTES / bytes_per_col;
    if (by_memory == 0) by_memory = 1;

    size_t batch = std::clamp(by_memory, MIN_BATCH, MAX_BATCH);

    // Keep enough work granularity for threading.
    const size_t min_for_threads = std::max<size_t>(MIN_BATCH, requested_threads * 8);
    if (batch < min_for_threads) {
        batch = std::min(MAX_BATCH, min_for_threads);
    }

    // Round down to cache-friendly multiple.
    if (batch >= 32) {
        batch = (batch / 32) * 32;
    }
    return std::max<size_t>(1, batch);
}

// Fix #2: Two-phase full-matrix merge to avoid data races on shared distance_arr.
// Phase 1 computes updated columns in parallel (read-only), phase 2 writes back serially.
// Merges are processed in adaptive batches to cap peak memory — each batch pre-allocates
// at most MERGE_BATCH columns instead of one per merge.
// Cross-batch interactions are handled by the updated dist; intra-batch by the patch.
void update_cluster_dissimilarities(
    std::vector<std::pair<int, int> >& merges,
    std::vector<Cluster>& clusters,
    SymDistMatrix& dist,
    const int NO_PROCESSORS,
    std::vector<int>& dsu_parent,
    std::vector<int>& dsu_size,
    std::vector<SymDistVector>& merged_columns_workspace,
    std::vector<char>& is_iter_secondary_workspace,
    double max_merge_distance,
    const std::vector<char>& is_alive_ws) {

    if (merges.empty()) {
        return;
    }
    const bool profile_ops = racpp_profile_ops_enabled();

    const size_t merge_count = merges.size();
    const int N = dist.N;
    const size_t requested_threads = (NO_PROCESSORS > 0) ? static_cast<size_t>(NO_PROCESSORS) : 1;
    const size_t MERGE_BATCH = choose_merge_batch_size(N, requested_threads);

    auto& merged_columns = merged_columns_workspace;
    const size_t max_batch = std::min(merge_count, MERGE_BATCH);
    if (merged_columns.size() < max_batch) {
        merged_columns.resize(max_batch);
    }
    for (size_t i = 0; i < max_batch; i++) {
        if (merged_columns[i].size() != N) {
            merged_columns[i].resize(N);
        }
    }

    // Reused mask for secondaries in this whole merge iteration.
    // Needed because last-batch main NN refresh must skip all secondaries,
    // including those from earlier batches.
    auto& is_iter_secondary = is_iter_secondary_workspace;
    if (static_cast<int>(is_iter_secondary.size()) < N) {
        is_iter_secondary.resize(N, 0);
    }
    for (const auto& merge : merges) {
        is_iter_secondary[merge.second] = 1;
    }

    // Alive ids needed for distance compute/write-back in each sub-batch.
    // Keep ids in ascending order so candidate iteration is contiguous/stable.
    std::vector<int> write_candidates;
    write_candidates.reserve(static_cast<size_t>(N));
    std::vector<char> is_processed_secondary(static_cast<size_t>(N), 0);
    std::vector<char> is_batch_main(static_cast<size_t>(N), 0);
    // NN refresh candidates (exclude iteration secondaries).
    std::vector<int> nn_candidates;
    nn_candidates.reserve(static_cast<size_t>(N));
    for (int k = 0; k < N; ++k) {
        if (!is_alive_ws[k]) continue;
        if (!is_iter_secondary[k]) {
            nn_candidates.push_back(k);
        }
    }
    struct IdRun {
        int k_start;
        size_t len;
    };
    std::vector<IdRun> nn_candidate_runs;
    nn_candidate_runs.reserve(nn_candidates.size() > 0 ? nn_candidates.size() / 4 : 0);
    if (!nn_candidates.empty()) {
        size_t run_start = 0;
        const size_t nn_count = nn_candidates.size();
        for (size_t c = 1; c <= nn_count; ++c) {
            const bool is_break =
                (c == nn_count) ||
                (nn_candidates[c] != nn_candidates[c - 1] + 1);
            if (!is_break) continue;
            nn_candidate_runs.push_back(IdRun{
                nn_candidates[run_start],
                c - run_start
            });
            run_start = c;
        }
    }
    const IdRun* nn_run_data = nn_candidate_runs.data();
    const size_t nn_run_count = nn_candidate_runs.size();

    // Reuse per-batch metadata buffers across sub-batches.
    std::vector<int> merge_main_ids(max_batch);
    std::vector<int> merge_secondary_ids(max_batch);
    std::vector<double> merge_main_sizes(max_batch);
    std::vector<double> merge_secondary_sizes(max_batch);
    std::vector<double> merge_inv_sizes(max_batch);
    std::vector<SymDistScalar> cross_dist(
        std::max<size_t>(1, max_batch * (max_batch - 1) / 2),
        std::numeric_limits<SymDistScalar>::infinity());
    std::vector<size_t> cross_row_start(max_batch, 0);

    for (size_t batch_start = 0; batch_start < merge_count; batch_start += MERGE_BATCH) {
        const size_t batch_end = std::min(batch_start + MERGE_BATCH, merge_count);
        const size_t batch_size = batch_end - batch_start;

        for (size_t i = 0; i < batch_size; i++) {
            const size_t global_i = batch_start + i;
            merge_main_ids[i] = merges[global_i].first;
            merge_secondary_ids[i] = merges[global_i].second;
            merge_main_sizes[i] = static_cast<double>(dsu_size[merge_main_ids[i]]);
            merge_secondary_sizes[i] = static_cast<double>(dsu_size[merge_secondary_ids[i]]);
            merge_inv_sizes[i] = 1.0 / (merge_main_sizes[i] + merge_secondary_sizes[i]);
            is_batch_main[static_cast<size_t>(merge_main_ids[i])] = 1;
        }

        // Precompute triangular row bases for cross_dist(i,j), i<j.
        for (size_t i = 0; i < batch_size; ++i) {
            cross_row_start[i] =
                i * batch_size - (i * (i + 1)) / 2;
        }

        auto precompute_cross_range = [&](size_t start, size_t end) {
            for (size_t i = start; i < end; i++) {
                const int mi = merge_main_ids[i];
                const int si = merge_secondary_ids[i];
                const double sz_mi = merge_main_sizes[i];
                const double sz_si = merge_secondary_sizes[i];
                const double inv_i = merge_inv_sizes[i];
                for (size_t j = i + 1; j < batch_size; j++) {
                    const int mj = merge_main_ids[j];
                    const int sj = merge_secondary_ids[j];
                    const double sz_mj = merge_main_sizes[j];
                    const double sz_sj = merge_secondary_sizes[j];
                    const double inv_j = merge_inv_sizes[j];
                    const double d_mi_mj = dist.get(mi, mj);
                    const double d_mi_sj = dist.get(mi, sj);
                    const double d_si_mj = dist.get(si, mj);
                    const double d_si_sj = dist.get(si, sj);
                    const double d_mi_to_j = (sz_mj * d_mi_mj + sz_sj * d_mi_sj) * inv_j;
                    const double d_si_to_j = (sz_mj * d_si_mj + sz_sj * d_si_sj) * inv_j;
                    const double d_ij = (sz_mi * d_mi_to_j + sz_si * d_si_to_j) * inv_i;
                    const SymDistScalar val = static_cast<SymDistScalar>(d_ij);
                    cross_dist[cross_row_start[i] + (j - i - 1)] = val;
                }
            }
        };

        const size_t cross_threads = std::min(requested_threads, batch_size);
        const auto t_cross_0 = profile_ops ? std::chrono::high_resolution_clock::now()
                                           : std::chrono::high_resolution_clock::time_point{};
        if (cross_threads <= 1 || batch_size < 128) {
            precompute_cross_range(0, batch_size);
        } else {
            run_parallel_for(requested_threads, batch_size, precompute_cross_range);
        }
        if (profile_ops) {
            const auto t_cross_1 = std::chrono::high_resolution_clock::now();
            g_dense_ops_profile.dissim_cross_ns += ns_between(t_cross_0, t_cross_1);
        }

        const auto t_cand_0 = profile_ops ? std::chrono::high_resolution_clock::now()
                                          : std::chrono::high_resolution_clock::time_point{};
        write_candidates.clear();
        for (int k = 0; k < N; ++k) {
            if (is_alive_ws[k] && !is_processed_secondary[static_cast<size_t>(k)]) {
                write_candidates.push_back(k);
            }
        }

        const int* write_cand_data = write_candidates.data();
        const size_t write_cand_count = write_candidates.size();
        struct CandidateRun {
            int k_start;
            int k_end;
            size_t len;
        };
        std::vector<CandidateRun> write_candidate_runs;
        write_candidate_runs.reserve(write_cand_count > 0 ? write_cand_count / 4 : 0);
        if (write_cand_count > 0) {
            size_t run_start = 0;
            for (size_t c = 1; c <= write_cand_count; ++c) {
                const bool is_break =
                    (c == write_cand_count) ||
                    (write_cand_data[c] != write_cand_data[c - 1] + 1);
                if (!is_break) continue;
                const int k0 = write_cand_data[run_start];
                const int k1 = write_cand_data[c - 1];
                write_candidate_runs.push_back(
                    CandidateRun{k0, k1, c - run_start});
                run_start = c;
            }
        }
        const CandidateRun* run_data = write_candidate_runs.data();
        const size_t run_count = write_candidate_runs.size();
        std::vector<CandidateRun> write_non_main_runs;
        write_non_main_runs.reserve(run_count);
        for (size_t r = 0; r < run_count; ++r) {
            const CandidateRun& run = run_data[r];
            int seg_start = -1;
            for (int k = run.k_start; k <= run.k_end; ++k) {
                if (!is_batch_main[static_cast<size_t>(k)]) {
                    if (seg_start < 0) seg_start = k;
                    continue;
                }
                if (seg_start >= 0) {
                    const int seg_end = k - 1;
                    write_non_main_runs.push_back(
                        CandidateRun{
                            seg_start,
                            seg_end,
                            static_cast<size_t>(seg_end - seg_start + 1)
                        });
                    seg_start = -1;
                }
            }
            if (seg_start >= 0) {
                write_non_main_runs.push_back(
                    CandidateRun{
                        seg_start,
                        run.k_end,
                        static_cast<size_t>(run.k_end - seg_start + 1)
                    });
            }
        }
        const CandidateRun* non_main_run_data = write_non_main_runs.data();
        const size_t non_main_run_count = write_non_main_runs.size();
        if (profile_ops) {
            const auto t_cand_1 = std::chrono::high_resolution_clock::now();
            g_dense_ops_profile.dissim_candidate_build_ns += ns_between(t_cand_0, t_cand_1);
        }

        // Parallel column computation over live candidates only.
        auto compute_range = [&](size_t start, size_t end) {
            const SymDistScalar inf = std::numeric_limits<SymDistScalar>::infinity();
            const size_t* row_start = dist.row_start.data();
            const SymDistScalar* dist_data = dist.data.data();

            for (size_t i = start; i < end; i++) {
                const int main_id = merge_main_ids[i];
                const int secondary_id = merge_secondary_ids[i];
                const SymDistScalar main_weight =
                    static_cast<SymDistScalar>(merge_main_sizes[i] * merge_inv_sizes[i]);
                const SymDistScalar secondary_weight =
                    static_cast<SymDistScalar>(merge_secondary_sizes[i] * merge_inv_sizes[i]);
                SymDistVector& out_col = merged_columns[i];

                const size_t main_base = row_start[static_cast<size_t>(main_id)];
                const size_t sec_base = row_start[static_cast<size_t>(secondary_id)];
                const int cutoff_id = std::max(main_id, secondary_id);
                const int* tail_begin = std::upper_bound(
                    write_cand_data, write_cand_data + write_cand_count, cutoff_id);
                const size_t tail_start =
                    static_cast<size_t>(tail_begin - write_cand_data);

                // Head region: at least one source is scattered in triangular storage.
                for (size_t c = 0; c < tail_start; ++c) {
                    const int k = write_cand_data[c];
                    if (k == main_id || k == secondary_id) {
                        out_col[k] = inf;
                        continue;
                    }

                    const SymDistScalar d_main = (k < main_id)
                        ? dist_data[row_start[static_cast<size_t>(k)] +
                                    static_cast<size_t>(main_id - k - 1)]
                        : dist_data[main_base +
                                    static_cast<size_t>(k - main_id - 1)];
                    const SymDistScalar d_sec = (k < secondary_id)
                        ? dist_data[row_start[static_cast<size_t>(k)] +
                                    static_cast<size_t>(secondary_id - k - 1)]
                        : dist_data[sec_base +
                                    static_cast<size_t>(k - secondary_id - 1)];
                    out_col[k] = main_weight * d_main + secondary_weight * d_sec;
                }

                // Tail region: k > max(main_id, secondary_id).
                // For contiguous candidate runs, both source rows and destination are contiguous.
                // Runs are precomputed once per sub-batch to avoid per-merge run detection cost.
                size_t run_idx = 0;
                size_t lo = 0, hi = run_count;
                while (lo < hi) {
                    const size_t mid = lo + (hi - lo) / 2;
                    if (run_data[mid].k_end <= cutoff_id) {
                        lo = mid + 1;
                    } else {
                        hi = mid;
                    }
                }
                run_idx = lo;
                while (run_idx < run_count) {
                    const CandidateRun& run = run_data[run_idx];
                    size_t run_off = 0;
                    if (cutoff_id >= run.k_start) {
                        run_off = static_cast<size_t>(cutoff_id - run.k_start + 1);
                        if (run_off >= run.len) {
                            ++run_idx;
                            continue;
                        }
                    }
                    const int k0 = run.k_start + static_cast<int>(run_off);
                    const size_t run_len = run.len - run_off;
                    const SymDistScalar* main_ptr =
                        dist_data + main_base + static_cast<size_t>(k0 - main_id - 1);
                    const SymDistScalar* sec_ptr =
                        dist_data + sec_base + static_cast<size_t>(k0 - secondary_id - 1);
                    SymDistScalar* out_ptr = out_col.data() + static_cast<size_t>(k0);

#if defined(RACPP_SIMD_TAIL_UPDATE) && RACPP_SIMD_TAIL_UPDATE && \
    defined(__AVX2__) && defined(__FMA__) && \
    defined(RACPP_SYMDIST_USE_FLOAT) && RACPP_SYMDIST_USE_FLOAT
                    size_t off = 0;
                    const __m256 mw_vec = _mm256_set1_ps(main_weight);
                    const __m256 sw_vec = _mm256_set1_ps(secondary_weight);
                    for (; off + 7 < run_len; off += 8) {
                        const __m256 dm = _mm256_loadu_ps(main_ptr + off);
                        const __m256 ds = _mm256_loadu_ps(sec_ptr + off);
                        const __m256 out =
                            _mm256_fmadd_ps(mw_vec, dm, _mm256_mul_ps(sw_vec, ds));
                        _mm256_storeu_ps(out_ptr + off, out);
                    }
                    for (; off < run_len; ++off) {
                        out_ptr[off] = main_weight * main_ptr[off] + secondary_weight * sec_ptr[off];
                    }
#else
                    for (size_t off = 0; off < run_len; ++off) {
                        out_ptr[off] = main_weight * main_ptr[off] + secondary_weight * sec_ptr[off];
                    }
#endif
                    ++run_idx;
                }

                for (size_t j = 0; j < i; ++j) {
                    const SymDistScalar patched =
                        cross_dist[cross_row_start[j] + (i - j - 1)];
                    out_col[merge_main_ids[j]] = patched;
                }
                for (size_t j = i + 1; j < batch_size; ++j) {
                    const SymDistScalar patched =
                        cross_dist[cross_row_start[i] + (j - i - 1)];
                    out_col[merge_main_ids[j]] = patched;
                }

                out_col[main_id] = inf;
                out_col[secondary_id] = inf;
            }
        };

        size_t no_threads = std::min(requested_threads, batch_size);
        const auto t_compute_0 = profile_ops ? std::chrono::high_resolution_clock::now()
                                             : std::chrono::high_resolution_clock::time_point{};
        if (no_threads <= 1) {
            compute_range(0, batch_size);
        } else {
            run_parallel_for(requested_threads, batch_size, compute_range);
        }
        if (profile_ops) {
            const auto t_compute_1 = std::chrono::high_resolution_clock::now();
            g_dense_ops_profile.dissim_compute_ns += ns_between(t_compute_0, t_compute_1);
        }

        // Write-back for merge mains:
        // 1) parallel non-main writes per merge main (disjoint address sets)
        // 2) deduplicated main-main writes once per pair
        // 3) serial DSU unions
        const auto t_write_0 = profile_ops ? std::chrono::high_resolution_clock::now()
                                           : std::chrono::high_resolution_clock::time_point{};
        auto set_col_active_runs = [&](int col_id, const SymDistVector& col) {
            const size_t* row_start = dist.row_start.data();
            const size_t col_base = row_start[static_cast<size_t>(col_id)];
            for (size_t r = 0; r < non_main_run_count; ++r) {
                const CandidateRun& run = non_main_run_data[r];
                if (run.k_end < col_id) {
                    for (int k = run.k_start; k <= run.k_end; ++k) {
                        const size_t idx =
                            row_start[static_cast<size_t>(k)] +
                            static_cast<size_t>(col_id - k - 1);
                        dist.data[idx] = col[k];
                    }
                    continue;
                }

                if (run.k_start > col_id) {
                    const int k0 = run.k_start;
                    SymDistScalar* dst =
                        dist.data.data() + col_base + static_cast<size_t>(k0 - col_id - 1);
                    const SymDistScalar* src = col.data() + static_cast<size_t>(k0);
                    std::copy_n(src, run.len, dst);
                    continue;
                }

                // Run intersects col_id: write left scattered and right contiguous.
                for (int k = run.k_start; k < col_id; ++k) {
                    const size_t idx =
                        row_start[static_cast<size_t>(k)] +
                        static_cast<size_t>(col_id - k - 1);
                    dist.data[idx] = col[k];
                }
                if (col_id < run.k_end) {
                    const int k0 = col_id + 1;
                    const size_t len = static_cast<size_t>(run.k_end - col_id);
                    SymDistScalar* dst =
                        dist.data.data() + col_base + static_cast<size_t>(k0 - col_id - 1);
                    const SymDistScalar* src = col.data() + static_cast<size_t>(k0);
                    std::copy_n(src, len, dst);
                }
            }
        };
        auto write_non_main_range = [&](size_t start, size_t end) {
            for (size_t i = start; i < end; ++i) {
                set_col_active_runs(merge_main_ids[i], merged_columns[i]);
            }
        };
        const size_t write_threads = std::min(requested_threads, batch_size);
        if (write_threads <= 1 || batch_size < 64) {
            write_non_main_range(0, batch_size);
        } else {
            run_parallel_for(requested_threads, batch_size, write_non_main_range);
        }

        // Main-main pairs are duplicated across columns; write each pair once.
        auto write_main_pairs_range = [&](size_t start, size_t end) {
            const size_t* row_start = dist.row_start.data();
            SymDistScalar* dist_data = dist.data.data();
            for (size_t i = start; i < end; ++i) {
                const int mi = merge_main_ids[i];
                for (size_t j = i + 1; j < batch_size; ++j) {
                    const int mj = merge_main_ids[j];
                    const SymDistScalar v =
                        cross_dist[cross_row_start[i] + (j - i - 1)];
                    const int a = (mi < mj) ? mi : mj;
                    const int b = (mi < mj) ? mj : mi;
                    const size_t idx =
                        row_start[static_cast<size_t>(a)] +
                        static_cast<size_t>(b - a - 1);
                    dist_data[idx] = v;
                }
            }
        };
        if (write_threads <= 1 || batch_size < 64) {
            write_main_pairs_range(0, batch_size);
        } else {
            run_parallel_for(requested_threads, batch_size, write_main_pairs_range);
        }

        for (size_t i = 0; i < batch_size; ++i) {
            dsu_union(dsu_parent, dsu_size, merge_main_ids[i], merge_secondary_ids[i]);
        }
        if (profile_ops) {
            const auto t_write_1 = std::chrono::high_resolution_clock::now();
            g_dense_ops_profile.dissim_writeback_ns += ns_between(t_write_0, t_write_1);
        }

        // Mark processed secondaries so future sub-batches exclude them.
        for (size_t i = 0; i < batch_size; ++i) {
            const int sid = merge_secondary_ids[i];
            is_processed_secondary[static_cast<size_t>(sid)] = 1;
            is_batch_main[static_cast<size_t>(merge_main_ids[i])] = 0;
        }
        // ITEM 6: Compute NN for merge mains from contiguous merged_columns,
        // but ONLY for the last batch. Earlier batches' mains get stale NNs
        // because later batches modify distances — those are handled by
        // update_cluster_nn_dist instead.
        const bool is_last_batch = (batch_end >= merge_count);
        if (is_last_batch) {
            const auto t_last_nn_0 = profile_ops ? std::chrono::high_resolution_clock::now()
                                                 : std::chrono::high_resolution_clock::time_point{};
            for (size_t i = 0; i < batch_size; i++) {
                const int main_id = merge_main_ids[i];
                const SymDistScalar* col_data = merged_columns[i].data();
                SymDistScalar best_val = std::numeric_limits<SymDistScalar>::infinity();
                int best_idx = -1;

                for (size_t r = 0; r < nn_run_count; ++r) {
                    const int k0 = nn_run_data[r].k_start;
                    const size_t len = nn_run_data[r].len;
                    const SymDistScalar* seg = col_data + static_cast<size_t>(k0);
                    for (size_t off = 0; off < len; ++off) {
                        const SymDistScalar v = seg[off];
                        if (v < best_val) {
                            best_val = v;
                            best_idx = k0 + static_cast<int>(off);
                        }
                    }
                }

                const double best_val_d = static_cast<double>(best_val);
                clusters[main_id].nn = (best_val_d < max_merge_distance) ? best_idx : -1;
                clusters[main_id].nn_distance = best_val_d;
            }
            if (profile_ops) {
                const auto t_last_nn_1 = std::chrono::high_resolution_clock::now();
                g_dense_ops_profile.dissim_last_batch_nn_ns +=
                    ns_between(t_last_nn_0, t_last_nn_1);
            }
        }
    }

    for (const auto& merge : merges) {
        is_iter_secondary[merge.second] = 0;
    }
}

template <typename Scalar>
static SymDistMatrix calculate_initial_dissimilarities_dense(
    Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>& base_arr,
    std::vector<Cluster>& clusters,
    double max_merge_distance,
    const std::string& distance_metric) {

    const int N = static_cast<int>(clusters.size());
    const int D = static_cast<int>(base_arr.rows());
    const int TILE = 1024;
    const bool is_cosine = (distance_metric == "cosine");
    const size_t n_threads = std::max<size_t>(1, static_cast<size_t>(Eigen::nbThreads()));

    SymDistMatrix dist(N);

    Eigen::Matrix<Scalar, 1, Eigen::Dynamic> sq_norms;
    if (!is_cosine) {
        sq_norms = base_arr.colwise().squaredNorm();
    }

    // Build tile pair worklist for parallel dispatch.
    struct TilePair { int i_start, i_end, j_start, j_end; };
    std::vector<TilePair> tile_pairs;
    tile_pairs.reserve(
        ((N + TILE - 1) / TILE) * ((N + TILE - 1) / TILE + 1) / 2);
    for (int i_start = 0; i_start < N; i_start += TILE) {
        const int i_end = std::min(i_start + TILE, N);
        for (int j_start = i_start; j_start < N; j_start += TILE) {
            const int j_end = std::min(j_start + TILE, N);
            tile_pairs.push_back({i_start, i_end, j_start, j_end});
        }
    }

    // Phase 1: Parallel tile GEMM + storage + thread-local NN tracking.
    // Each tile pair writes to a disjoint region of dist.data, so no races.
    // Single-threaded GEMM per tile avoids Eigen thread sync overhead.
    const size_t local_slots = std::min(n_threads, tile_pairs.size());
    const double inf = std::numeric_limits<double>::infinity();
    std::vector<std::vector<double>> nn_best_locals(
        local_slots, std::vector<double>(N, inf));
    std::vector<std::vector<int>> nn_idx_locals(
        local_slots, std::vector<int>(N, -1));
    std::atomic<size_t> slot_counter{0};

    const int saved_threads = Eigen::nbThreads();
    Eigen::setNbThreads(1);

    run_parallel_for(n_threads, tile_pairs.size(), [&](size_t start, size_t end) {
        const size_t slot = slot_counter.fetch_add(1, std::memory_order_relaxed);
        std::vector<double>& nn_best_local = nn_best_locals[slot];
        std::vector<int>& nn_idx_local = nn_idx_locals[slot];

        auto update_local_nn = [&](int src, int dst, double val) {
            double& best = nn_best_local[src];
            int& idx = nn_idx_local[src];
            if (val < best || (val == best && (idx == -1 || dst < idx))) {
                best = val;
                idx = dst;
            }
        };

        using TileMatrix =
            Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
        TileMatrix tile;

        for (size_t t = start; t < end; t++) {
            const auto& tp = tile_pairs[t];
            const int tile_i = tp.i_end - tp.i_start;
            const int tile_j = tp.j_end - tp.j_start;

            auto Bi = base_arr.block(0, tp.i_start, D, tile_i);
            auto Bj = base_arr.block(0, tp.j_start, D, tile_j);

            tile.noalias() = Bi.transpose() * Bj;

            if (!is_cosine) {
                tile *= Scalar(-2);
                for (int r = 0; r < tile_i; r++)
                    tile.row(r).array() += sq_norms[tp.i_start + r];
                for (int c = 0; c < tile_j; c++)
                    tile.col(c).array() += sq_norms[tp.j_start + c];
                tile = tile.array().max(Scalar(0)).sqrt().matrix();
            }

            // Store to triangular matrix — branchless, no NN tracking.
            if (tp.i_start == tp.j_start) {
                // Diagonal tile: upper triangle only.
                for (int r = 0; r < tile_i; r++) {
                    const int i_global = tp.i_start + r;
                    const int c_start = r + 1;
                    if (c_start >= tile_j) continue;
                    const size_t base_idx =
                        dist.row_start[static_cast<size_t>(i_global)];
                    if constexpr (std::is_same_v<Scalar, SymDistScalar>) {
                        for (int c = c_start; c < tile_j; c++) {
                            const Scalar v =
                                is_cosine ? (Scalar(1) - tile(r, c)) : tile(r, c);
                            dist.data[base_idx + (c - c_start)] = v;
                            const int j_global = tp.j_start + c;
                            const double val = static_cast<double>(v);
                            update_local_nn(i_global, j_global, val);
                            update_local_nn(j_global, i_global, val);
                        }
                    } else {
                        for (int c = c_start; c < tile_j; c++) {
                            const Scalar v =
                                is_cosine ? (Scalar(1) - tile(r, c)) : tile(r, c);
                            dist.data[base_idx + (c - c_start)] =
                                static_cast<SymDistScalar>(v);
                            const int j_global = tp.j_start + c;
                            const double val = static_cast<double>(v);
                            update_local_nn(i_global, j_global, val);
                            update_local_nn(j_global, i_global, val);
                        }
                    }
                }
            } else {
                // Off-diagonal tile: all (i,j) satisfy i < j.
                // Writes are contiguous per row in triangular storage.
                for (int r = 0; r < tile_i; r++) {
                    const int i_global = tp.i_start + r;
                    const size_t base_idx =
                        dist.row_start[static_cast<size_t>(i_global)] +
                        static_cast<size_t>(tp.j_start - i_global - 1);
                    if constexpr (std::is_same_v<Scalar, SymDistScalar>) {
#if defined(RACPP_SPLIT_STORE_NN) && RACPP_SPLIT_STORE_NN
                        for (int c = 0; c < tile_j; c++) {
                            const Scalar v =
                                is_cosine ? (Scalar(1) - tile(r, c)) : tile(r, c);
                            dist.data[base_idx + c] = v;
                        }
                        const SymDistScalar* written = dist.data.data() + base_idx;
                        for (int c = 0; c < tile_j; c++) {
                            const int j_global = tp.j_start + c;
                            const double val = static_cast<double>(written[c]);
                            update_local_nn(i_global, j_global, val);
                            update_local_nn(j_global, i_global, val);
                        }
#else
                        for (int c = 0; c < tile_j; c++) {
                            const Scalar v =
                                is_cosine ? (Scalar(1) - tile(r, c)) : tile(r, c);
                            dist.data[base_idx + c] = v;
                            const int j_global = tp.j_start + c;
                            const double val = static_cast<double>(v);
                            update_local_nn(i_global, j_global, val);
                            update_local_nn(j_global, i_global, val);
                        }
#endif
                    } else {
#if defined(RACPP_SPLIT_STORE_NN) && RACPP_SPLIT_STORE_NN
                        for (int c = 0; c < tile_j; c++) {
                            const Scalar v =
                                is_cosine ? (Scalar(1) - tile(r, c)) : tile(r, c);
                            dist.data[base_idx + c] =
                                static_cast<SymDistScalar>(v);
                        }
                        const SymDistScalar* written = dist.data.data() + base_idx;
                        for (int c = 0; c < tile_j; c++) {
                            const int j_global = tp.j_start + c;
                            const double val = static_cast<double>(written[c]);
                            update_local_nn(i_global, j_global, val);
                            update_local_nn(j_global, i_global, val);
                        }
#else
                        for (int c = 0; c < tile_j; c++) {
                            const Scalar v =
                                is_cosine ? (Scalar(1) - tile(r, c)) : tile(r, c);
                            dist.data[base_idx + c] =
                                static_cast<SymDistScalar>(v);
                            const int j_global = tp.j_start + c;
                            const double val = static_cast<double>(v);
                            update_local_nn(i_global, j_global, val);
                            update_local_nn(j_global, i_global, val);
                        }
#endif
                    }
                }
            }
        }
    });

    Eigen::setNbThreads(saved_threads);

    // Reduce thread-local NN tracking deterministically.
    std::vector<double> nn_best(N, inf);
    std::vector<int> nn_idx(N, -1);
    for (size_t s = 0; s < local_slots; s++) {
        const std::vector<double>& local_best = nn_best_locals[s];
        const std::vector<int>& local_idx = nn_idx_locals[s];
        for (int k = 0; k < N; k++) {
            const int idx = local_idx[k];
            if (idx < 0) continue;
            const double val = local_best[k];
            if (val < nn_best[k] || (val == nn_best[k] && (nn_idx[k] == -1 || idx < nn_idx[k]))) {
                nn_best[k] = val;
                nn_idx[k] = idx;
            }
        }
    }

    for (int k = 0; k < N; k++) {
        clusters[k].nn = (nn_best[k] < max_merge_distance) ? nn_idx[k] : -1;
        clusters[k].nn_distance = nn_best[k];
    }

    return dist;
}

SymDistMatrix calculate_initial_dissimilarities(
    Eigen::MatrixXd& base_arr,
    std::vector<Cluster>& clusters,
    double max_merge_distance,
    std::string distance_metric) {
    return calculate_initial_dissimilarities_dense(base_arr, clusters, max_merge_distance, distance_metric);
}

void calculate_initial_dissimilarities(
    Eigen::MatrixXd& base_arr,
    std::vector<Cluster>& clusters,
    Eigen::SparseMatrix<bool>& connectivity,
    double max_merge_distance,
    int batch_size,
    std::string distance_metric) {

    const bool is_cosine = (distance_metric == "cosine");
    const int clustersSize = static_cast<int>(clusters.size());

    Eigen::VectorXd sq_norms;
    if (!is_cosine) {
        sq_norms = base_arr.colwise().squaredNorm();
    }

    const size_t requested_threads =
        std::max<size_t>(1, static_cast<size_t>(Eigen::nbThreads()));

    auto process_cluster = [&](int i) {
        Cluster& cluster = clusters[i];
        auto base_col = base_arr.col(i);

        std::vector<std::pair<int, double>> neighbors;

        int nearest_neighbor = -1;
        double min = std::numeric_limits<double>::infinity();

        for (Eigen::SparseMatrix<bool>::InnerIterator it(connectivity, i); it; ++it) {
            int j = it.index();
            bool value = it.value();

            if (j != i && value) {
                const double dot = base_col.dot(base_arr.col(j));
                double distance = 0.0;
                if (is_cosine) {
                    distance = 1.0 - dot;
                } else {
                    double sq_dist = sq_norms[i] + sq_norms[j] - 2.0 * dot;
                    if (sq_dist < 0.0) {
                        sq_dist = 0.0;
                    }
                    distance = std::sqrt(sq_dist);
                }

                neighbors.push_back(std::make_pair(j, distance));

                if (distance < min && distance < max_merge_distance) {
                    min = distance;
                    nearest_neighbor = j;
                }
            }
        }

        cluster.neighbor_distances = std::move(neighbors);
        cluster.nn = nearest_neighbor;
        cluster.nn_distance = min;
    };

    for (int batchStart = 0; batchStart < clustersSize; batchStart += batch_size) {
        int batchEnd = std::min(batchStart + batch_size, clustersSize);
        const int batchCount = batchEnd - batchStart;
        const size_t no_threads = std::min(requested_threads, static_cast<size_t>(batchCount));

        if (no_threads <= 1 || batchCount < 64) {
            for (int i = batchStart; i < batchEnd; ++i) {
                process_cluster(i);
            }
            continue;
        }

        std::vector<std::thread> threads;
        threads.reserve(no_threads);

        size_t chunk_size = static_cast<size_t>(batchCount) / no_threads;
        size_t remainder = static_cast<size_t>(batchCount) % no_threads;
        int start = batchStart;
        for (size_t t = 0; t < no_threads; t++) {
            int end = start + static_cast<int>(chunk_size) + (t < remainder ? 1 : 0);
            threads.emplace_back([&process_cluster, start, end]() {
                for (int i = start; i < end; ++i) {
                    process_cluster(i);
                }
            });
            start = end;
        }

        for (auto& thread : threads) {
            thread.join();
        }
    }
}

//-----------------------End Distance Calculations-------------------------

//-----------------------Merging Functions-----------------------------------
// Fix #3: Binary search in sorted dissimilarities vector instead of unordered_map lookup
double get_cluster_distances(
    Cluster& main_cluster,
    std::vector<int>& other_cluster_idxs,
    int other_cluster_id,
    Eigen::MatrixXd& base_arr) {

    // Binary search in sorted dissimilarities
    auto it = std::lower_bound(main_cluster.dissimilarities.begin(), main_cluster.dissimilarities.end(),
        other_cluster_id,
        [](const std::pair<int, double>& p, int key) { return p.first < key; });

    if (it != main_cluster.dissimilarities.end() && it->first == other_cluster_id) {
        return it->second;
    } else {
        const size_t size_main = main_cluster.indices.size();
        const size_t size_other = other_cluster_idxs.size();
        if (size_main == 0 || size_other == 0) {
            return std::numeric_limits<double>::infinity();
        }

        // mean(1 - dot(a, b)) over all a in A, b in B:
        // 1 - (sum_A · sum_B) / (|A| * |B|)
        Eigen::VectorXd sum_main = Eigen::VectorXd::Zero(base_arr.rows());
        Eigen::VectorXd sum_other = Eigen::VectorXd::Zero(base_arr.rows());

        for (int idx : main_cluster.indices) {
            sum_main.noalias() += base_arr.col(idx);
        }
        for (int idx : other_cluster_idxs) {
            sum_other.noalias() += base_arr.col(idx);
        }

        const double mean_dot = sum_main.dot(sum_other) / static_cast<double>(size_main * size_other);
        return 1.0 - mean_dot;
    }
}

std::pair<std::vector<int>, std::vector<int>> split_neighbors(
    Cluster& main_cluster,
    Cluster& secondary_cluster,
    std::vector<Cluster>& clusters,
    std::vector<int>& merging_array) {

    std::vector<int> static_neighbors;
    static_neighbors.reserve(main_cluster.neighbors.size() + secondary_cluster.neighbors.size());

    std::vector<int> merging_neighbors;
    merging_neighbors.reserve(main_cluster.neighbors.size() + secondary_cluster.neighbors.size());

    for (auto& id : main_cluster.neighbors) {
        if (id != main_cluster.id && id != secondary_cluster.id) {
            int smallest_id = id < clusters[id].nn ? id : clusters[id].nn;
            if (clusters[id].will_merge) {
                if (merging_array[smallest_id] == 0) {
                    merging_neighbors.push_back(smallest_id);
                }

                merging_array[smallest_id]++;
            } else {
                merging_array[id] = 1;
                static_neighbors.push_back(id);
            }
        }
    }

    for (auto& id : secondary_cluster.neighbors) {
        if (id != main_cluster.id && id != secondary_cluster.id) {
            int smallest_id = id < clusters[id].nn ? id : clusters[id].nn;

            if (clusters[id].will_merge) {
                if (merging_array[smallest_id] == 0) {
                    merging_neighbors.push_back(smallest_id);
                }
                merging_array[smallest_id]++;

            } else {
                if (merging_array[id] == 0) {
                    static_neighbors.push_back(id);
                }
                ++merging_array[id];
            }
        }
    }

    return std::make_pair(static_neighbors, merging_neighbors);
}

void merge_cluster_symmetric_linkage(
    std::pair<int, int>& merge,
    std::vector<Cluster>& clusters,
    std::vector<std::pair<int, double>>& merging_array) {

    Cluster& main_cluster = clusters[merge.first];
    Cluster& secondary_cluster = clusters[merge.second];

    std::vector<std::pair<int, double>> new_neighbors;
    std::vector<std::tuple<int, int, double>> needs_update;
    std::vector<int> unique_neighbors;

    // First loop through main neighbors
    for (auto& neighbor : main_cluster.neighbor_distances) {
        if (neighbor.first == main_cluster.id || neighbor.first == secondary_cluster.id) {
            continue;
        }

        merging_array[neighbor.first].first = main_cluster.id + 1;
        if (clusters[neighbor.first].will_merge) {
            merging_array[neighbor.first].second = (clusters[neighbor.first].indices.size() + main_cluster.indices.size()) * neighbor.second;
        } else {
            merging_array[neighbor.first].second = main_cluster.indices.size() * neighbor.second;
        }

        unique_neighbors.push_back(neighbor.first);
    }

    // Then loop through secondary neighbors
    for (auto& neighbor : secondary_cluster.neighbor_distances) {
        if (neighbor.first == main_cluster.id || neighbor.first == secondary_cluster.id) {
            continue;
        }

        if (merging_array[neighbor.first].first <= 0) {
            unique_neighbors.push_back(neighbor.first);
        }

        if (clusters[neighbor.first].will_merge) {
            if (merging_array[neighbor.first].first == main_cluster.id + 1) {
                merging_array[neighbor.first].second += (clusters[neighbor.first].indices.size() + secondary_cluster.indices.size()) * neighbor.second;
            } else {
                merging_array[neighbor.first].second = (clusters[neighbor.first].indices.size() + secondary_cluster.indices.size()) * neighbor.second;
            }

        } else {
            if (merging_array[neighbor.first].first == main_cluster.id + 1) {
                int indices_sum = main_cluster.indices.size() + secondary_cluster.indices.size();
                merging_array[neighbor.first].second = (merging_array[neighbor.first].second + secondary_cluster.indices.size() * neighbor.second) / indices_sum;
            } else {
                merging_array[neighbor.first].second = neighbor.second;
            }
        }
        merging_array[neighbor.first].first += secondary_cluster.id + 1;
    }

    for (auto& neighbor_id : unique_neighbors) {
        if (!clusters[neighbor_id].will_merge) {
            double new_dist = merging_array[neighbor_id].second;
            if (merging_array[neighbor_id].first == main_cluster.id + 1) {
                new_dist = new_dist / main_cluster.indices.size();
            }
            new_neighbors.push_back(std::make_pair(neighbor_id, new_dist));
            needs_update.push_back(std::make_tuple(main_cluster.id, neighbor_id, new_dist));
            merging_array[neighbor_id].first = 0;
            merging_array[neighbor_id].second = 0.0;
            continue;
        }

        if (clusters[neighbor_id].will_merge &&
        (merging_array[neighbor_id].first == -1 || merging_array[clusters[neighbor_id].nn].first == -1)) {
            int min_id = std::min(clusters[neighbor_id].nn, neighbor_id);
            merging_array[min_id].first = 0;
            merging_array[min_id].second = 0.0;
            continue;
        }

        int nn_id = clusters[neighbor_id].nn;
        double total = merging_array[neighbor_id].second + merging_array[nn_id].second;
        double denominator = 0.0;

        if (merging_array[neighbor_id].first == main_cluster.id + secondary_cluster.id + 2) {
            denominator = main_cluster.indices.size() + secondary_cluster.indices.size() + clusters[neighbor_id].indices.size() * 2;

        } else if (merging_array[neighbor_id].first == main_cluster.id + 1) {
            denominator = main_cluster.indices.size() + clusters[neighbor_id].indices.size();

        } else if (merging_array[neighbor_id].first == secondary_cluster.id + 1) {
            denominator = secondary_cluster.indices.size() + clusters[neighbor_id].indices.size();
        }

        if (merging_array[nn_id].first == main_cluster.id + secondary_cluster.id + 2) {
            denominator += main_cluster.indices.size() + secondary_cluster.indices.size() + clusters[nn_id].indices.size() * 2;

        } else if (merging_array[nn_id].first == main_cluster.id + 1) {
            denominator += main_cluster.indices.size() + clusters[nn_id].indices.size();

        } else if (merging_array[nn_id].first == secondary_cluster.id + 1) {
            denominator += secondary_cluster.indices.size() + clusters[nn_id].indices.size();
        }

        double avg_dist = total / denominator;
        int smallest_id = std::min(neighbor_id, nn_id);
        new_neighbors.push_back(std::make_pair(smallest_id, avg_dist));

        merging_array[neighbor_id].first = 0;
        merging_array[neighbor_id].second = 0.0;
        if (merging_array[nn_id].first != 0) { // nn is in unique neighbors
            merging_array[nn_id].first = 0;
            merging_array[nn_id].second = 0.0;
            merging_array[smallest_id].first = -1;
        }
    }

    main_cluster.neighbor_distances = new_neighbors;
    main_cluster.neighbors_needing_updates = needs_update;
}

// Computes missing edges on the fly for a more balanced tree
// Fix #3: dissimilarities changed from unordered_map to sorted vector
void merge_cluster_compute_linkage(
    std::pair<int, int>& merge,
    std::vector<Cluster>& clusters,
    std::vector<int>& merging_array,
    Eigen::MatrixXd& base_arr) {

    Cluster& main_cluster = clusters[merge.first];
    Cluster& secondary_cluster = clusters[merge.second];

    std::vector<int> new_neighbors;

    std::vector<std::pair<int, double>> new_dissimilarities;
    new_dissimilarities.reserve(main_cluster.dissimilarities.size() + secondary_cluster.dissimilarities.size());

    std::vector<int> static_neighbors;
    std::vector<int> merging_neighbors;
    std::tie(static_neighbors, merging_neighbors) = split_neighbors(main_cluster, secondary_cluster, clusters, merging_array);

    std::vector<std::tuple<int, int, double> > needs_update;
    for (auto& static_id : static_neighbors) {
        double main_dist = get_cluster_distances(main_cluster, clusters[static_id].indices, static_id, base_arr);
        double secondary_dist = get_cluster_distances(secondary_cluster, clusters[static_id].indices, static_id, base_arr);

        double avg_dist = (main_cluster.indices.size() * main_dist + secondary_cluster.indices.size() * secondary_dist) / (main_cluster.indices.size() + secondary_cluster.indices.size());

        needs_update.push_back(std::make_tuple(main_cluster.id, static_id, avg_dist));
        new_neighbors.push_back(static_id);
        new_dissimilarities.push_back({static_id, avg_dist});
        merging_array[static_id] = 0;
    }

    for (auto& merging_id : merging_neighbors) {
        double main_primary_dist = get_cluster_distances(main_cluster, clusters[merging_id].indices, merging_id, base_arr);
        double main_secondary_dist = get_cluster_distances(secondary_cluster, clusters[merging_id].indices, merging_id, base_arr);
        double main_avg_dist = (main_cluster.indices.size() * main_primary_dist + secondary_cluster.indices.size() * main_secondary_dist) / (main_cluster.indices.size() + secondary_cluster.indices.size());

        int secondary_merging_id = clusters[merging_id].nn;
        double secondary_primary_dist = get_cluster_distances(main_cluster, clusters[secondary_merging_id].indices, secondary_merging_id, base_arr);
        double secondary_secondary_dist = get_cluster_distances(secondary_cluster, clusters[secondary_merging_id].indices, secondary_merging_id, base_arr);
        double secondary_avg_dist = (main_cluster.indices.size() * secondary_primary_dist + secondary_cluster.indices.size() * secondary_secondary_dist) / (main_cluster.indices.size() + secondary_cluster.indices.size());

        double avg_dist = (clusters[merging_id].indices.size() * main_avg_dist + clusters[secondary_merging_id].indices.size() * secondary_avg_dist) / (clusters[merging_id].indices.size() + clusters[secondary_merging_id].indices.size());

        new_neighbors.push_back(merging_id);
        new_dissimilarities.push_back({merging_id, avg_dist});

        merging_array[merging_id] = 0;
    }

    // Sort the new dissimilarities by key for future binary search lookups
    std::sort(new_dissimilarities.begin(), new_dissimilarities.end());

    main_cluster.neighbors = new_neighbors;
    main_cluster.dissimilarities = std::move(new_dissimilarities);
    main_cluster.neighbors_needing_updates = needs_update;
}

void merge_clusters_symmetric(
    std::vector<std::pair<int, int> >& merges,
    std::vector<Cluster>& clusters,
    std::vector<std::pair<int, double>>& merging_array) {

    for (auto& merge : merges) {
        merge_cluster_symmetric_linkage(merge, clusters, merging_array);
    }
}

void merge_clusters_compute(
    std::vector<std::pair<int, int> >& merges,
    std::vector<Cluster>& clusters,
    std::vector<int>& merging_array,
    Eigen::MatrixXd& base_arr) {
    for (auto& merge : merges) {
        merge_cluster_compute_linkage(merge, clusters, merging_array, base_arr);
    }
}

void parallel_merge_clusters(
    std::vector<std::pair<int, int> >& merges,
    std::vector<Cluster>& clusters,
    size_t no_threads,
    std::vector<std::vector<std::pair<int, double>>>& merging_arrays) {

    const size_t total = merges.size();
    if (total == 0) {
        return;
    }

    const size_t requested_threads = (no_threads > 0) ? no_threads : 1;
    const size_t worker_count = std::min(requested_threads, total);

    if (worker_count <= 1) {
        merge_clusters_symmetric(merges, clusters, merging_arrays[0]);
        return;
    }

    std::vector<std::thread> threads;
    threads.reserve(worker_count);

    const size_t chunk_size = total / worker_count;
    const size_t remainder = total % worker_count;
    size_t start = 0;

    for (size_t t = 0; t < worker_count; t++) {
        const size_t end = start + chunk_size + (t < remainder ? 1 : 0);
        threads.emplace_back([&merges, &clusters, &merging_arrays, start, end, t]() {
            for (size_t i = start; i < end; i++) {
                merge_cluster_symmetric_linkage(merges[i], clusters, merging_arrays[t]);
            }
        });
        start = end;
    }

    for (auto& thread : threads) {
        thread.join();
    }
}

void parallel_merge_clusters(
    std::vector<std::pair<int, int> >& merges,
    std::vector<Cluster>& clusters,
    size_t no_threads,
    std::vector<std::vector<int>>& merging_arrays,
    Eigen::MatrixXd& base_arr) {

    const size_t total = merges.size();
    if (total == 0) {
        return;
    }

    const size_t requested_threads = (no_threads > 0) ? no_threads : 1;
    const size_t worker_count = std::min(requested_threads, total);

    if (worker_count <= 1) {
        merge_clusters_compute(merges, clusters, merging_arrays[0], base_arr);
        return;
    }

    std::vector<std::thread> threads;
    threads.reserve(worker_count);

    const size_t chunk_size = total / worker_count;
    const size_t remainder = total % worker_count;
    size_t start = 0;

    for (size_t t = 0; t < worker_count; t++) {
        const size_t end = start + chunk_size + (t < remainder ? 1 : 0);
        threads.emplace_back([&merges, &clusters, &merging_arrays, &base_arr, start, end, t]() {
            for (size_t i = start; i < end; i++) {
                merge_cluster_compute_linkage(merges[i], clusters, merging_arrays[t], base_arr);
            }
        });
        start = end;
    }

    for (auto& thread : threads) {
        thread.join();
    }
}
//-----------------------End Merging Functions-----------------------------------

//-----------------------Updating Nearest Neighbors-----------------------------------

void update_cluster_neighbors(
    std::pair<int, std::vector<std::pair<int, double> > >& update_chunk,
    std::vector<Cluster>& clusters,
    std::vector<int>& update_neighbors) {
    Cluster& other_cluster = clusters[update_chunk.first];

    std::vector<std::pair<int, double>> new_neighbors;
    std::vector<int> all_looped_neighbors;
    for (size_t i=0; i<update_chunk.second.size(); i++) {
        int neighbor_id = update_chunk.second[i].first;
        int neighbor_nn_id = clusters[neighbor_id].nn;
        double dissimilarity = update_chunk.second[i].second;

        update_neighbors[neighbor_id] = 1;
        update_neighbors[neighbor_nn_id] = -1;

        if (dissimilarity >= 0) {
            new_neighbors.push_back(std::make_pair(neighbor_id, dissimilarity));
        }

        all_looped_neighbors.push_back(neighbor_id);
        all_looped_neighbors.push_back(neighbor_nn_id);
    }

    for (size_t i=0; i<other_cluster.neighbor_distances.size(); i++) {
        int neighbor_id = other_cluster.neighbor_distances[i].first;
        if (update_neighbors[neighbor_id] == 0) {
            new_neighbors.push_back(other_cluster.neighbor_distances[i]);
            all_looped_neighbors.push_back(neighbor_id);
        }
    }

    for (size_t i=0; i<all_looped_neighbors.size(); i++) {
        update_neighbors[all_looped_neighbors[i]] = 0;
    }

    other_cluster.neighbor_distances = new_neighbors;
}

void update_cluster_neighbors(
    SymDistMatrix& dist,
    const std::vector<std::pair<int, int>>& merges
) {
    // Symmetric storage: col-to-row copy is a no-op.
    // Just deactivate secondary clusters by filling their entries with infinity.
    for (size_t i = 0; i < merges.size(); i++) {
        dist.fill_infinity(merges[i].second);
    }
}

void update_cluster_neighbors_p(
    std::vector<std::pair<int, std::vector<std::pair<int, double> > > >& updates,
    std::vector<Cluster>& clusters,
    std::vector<int>& neighbor_sort_arr,
    std::vector<int>& update_neighbors) {
    for (auto& update: updates) {
        update_cluster_neighbors(update, clusters, update_neighbors);
        neighbor_sort_arr[update.first] = -1;
    }
}

void parallel_update_clusters(
    std::vector<std::pair<int, std::vector<std::pair<int, double>>>>& updates,
    std::vector<Cluster>& clusters,
    std::vector<std::vector<int>>& update_neighbors_arrays,
    std::vector<int>& neighbor_sort_arr,
    size_t no_threads) {

    const size_t total = updates.size();
    if (total == 0) {
        return;
    }

    const size_t requested_threads = (no_threads > 0) ? no_threads : 1;
    const size_t worker_count = std::min(requested_threads, total);

    if (worker_count <= 1) {
        for (size_t i = 0; i < total; i++) {
            update_cluster_neighbors(updates[i], clusters, update_neighbors_arrays[0]);
            neighbor_sort_arr[updates[i].first] = -1;
        }
        return;
    }

    std::vector<std::thread> threads;
    threads.reserve(worker_count);

    const size_t chunk_size = total / worker_count;
    const size_t remainder = total % worker_count;
    size_t start = 0;

    for (size_t t = 0; t < worker_count; t++) {
        const size_t end = start + chunk_size + (t < remainder ? 1 : 0);
        threads.emplace_back([&updates, &clusters, &neighbor_sort_arr, &update_neighbors_arrays, start, end, t]() {
            for (size_t i = start; i < end; i++) {
                update_cluster_neighbors(updates[i], clusters, update_neighbors_arrays[t]);
                neighbor_sort_arr[updates[i].first] = -1;
            }
        });
        start = end;
    }

    for (auto& thread : threads) {
        thread.join();
    }
}

// Fix #1: Takes indices to update instead of a separate vector of Cluster pointers
void update_cluster_nn(
    std::vector<Cluster>& clusters,
    const std::vector<int>& indices_to_update,
    double max_merge_distance,
    std::vector<int>& nn_count) {
    for (int idx : indices_to_update) {
        clusters[idx].update_nn(max_merge_distance);
        nn_count[clusters[idx].id] = 0;
    }
}

// Fix #1: Uses active_indices to iterate only live clusters
// Now parallelized: each cluster's update_nn is independent (read-only on dist).
void update_cluster_nn_dist(
    std::vector<Cluster>& clusters,
    const std::vector<int>& active_indices,
    const SymDistMatrix& dist,
    double max_merge_distance,
    const std::vector<std::pair<int, int>>& merges,
    const int NO_PROCESSORS,
    const std::vector<char>& is_alive_ws,
    std::vector<char>& is_dead_ws,
    std::vector<char>& is_changed_ws) {

    if (merges.empty()) return;
    const bool profile_ops = racpp_profile_ops_enabled();

    const int N = dist.N;
    if (static_cast<int>(is_dead_ws.size()) < N) is_dead_ws.assign(N, 0);
    if (static_cast<int>(is_changed_ws.size()) < N) is_changed_ws.assign(N, 0);

    // Last-batch boundary (must match batching in update_cluster_dissimilarities).
    const size_t requested = (NO_PROCESSORS > 0) ? static_cast<size_t>(NO_PROCESSORS) : 1;
    const size_t MERGE_BATCH = choose_merge_batch_size(N, requested);
    const size_t last_batch_start =
        ((merges.size() - 1) / MERGE_BATCH) * MERGE_BATCH;

    // Mark all mains as changed=1, all secondaries as dead.
    for (const auto& m : merges) {
        is_changed_ws[m.first] = 1;
        is_dead_ws[m.second] = 1;
    }
    // Upgrade last-batch mains to changed=2 (NN fresh from write-back, skip).
    for (size_t i = last_batch_start; i < merges.size(); i++) {
        is_changed_ws[merges[i].first] = 2;
    }

    std::vector<int> changed_main_ids;
    changed_main_ids.reserve(merges.size());
    for (const auto& m : merges) {
        changed_main_ids.push_back(m.first);
    }
    // The changed-main shortlist is only worthwhile when it reliably avoids full scans.
    // On dense runs it can have near-zero hit rate, so disable it aggressively.
    constexpr size_t MAX_CHANGED_MAIN_SHORTLIST = 512;
    constexpr uint64_t SHORTLIST_DISABLE_MIN_ATTEMPTS = 4096;
    constexpr uint64_t SHORTLIST_DISABLE_CALL_MIN_ATTEMPTS = 1024;
    constexpr double SHORTLIST_DISABLE_MIN_HIT_RATE = 5e-3; // 0.5%
    static std::atomic<uint64_t> s_shortlist_attempts_total{0};
    static std::atomic<uint64_t> s_shortlist_hits_total{0};
    static std::atomic<int> s_shortlist_disabled{0};
    const bool shortlist_globally_disabled =
        (s_shortlist_disabled.load(std::memory_order_relaxed) != 0);
    bool use_changed_shortlist =
        !shortlist_globally_disabled &&
        (changed_main_ids.size() <= MAX_CHANGED_MAIN_SHORTLIST);
    struct ChangedMainRun {
        int k_start;
        int k_end;
        size_t len;
    };
    std::vector<ChangedMainRun> changed_main_runs;
    const ChangedMainRun* changed_run_data = nullptr;
    size_t changed_run_count = 0;
    const auto t_changed_prep_0 = profile_ops ? std::chrono::high_resolution_clock::now()
                                              : std::chrono::high_resolution_clock::time_point{};
    if (use_changed_shortlist) {
        std::vector<int> changed_main_live;
        changed_main_live.reserve(changed_main_ids.size());
        for (int k : changed_main_ids) {
            if (is_alive_ws[k] && !is_dead_ws[k]) {
                changed_main_live.push_back(k);
            }
        }
        std::sort(changed_main_live.begin(), changed_main_live.end());
        changed_main_live.erase(
            std::unique(changed_main_live.begin(), changed_main_live.end()),
            changed_main_live.end());
        changed_main_runs.reserve(
            changed_main_live.size() > 0 ? changed_main_live.size() / 4 : 0);
        if (!changed_main_live.empty()) {
            size_t run_start = 0;
            const size_t count = changed_main_live.size();
            for (size_t c = 1; c <= count; ++c) {
                const bool is_break =
                    (c == count) ||
                    (changed_main_live[c] != changed_main_live[c - 1] + 1);
                if (!is_break) continue;
                const int k0 = changed_main_live[run_start];
                const int k1 = changed_main_live[c - 1];
                changed_main_runs.push_back(
                    ChangedMainRun{k0, k1, c - run_start});
                run_start = c;
            }
        }
        changed_run_data = changed_main_runs.data();
        changed_run_count = changed_main_runs.size();
    }
    if (profile_ops) {
        const auto t_changed_prep_1 = std::chrono::high_resolution_clock::now();
        g_dense_ops_profile.nn_changed_shortlist_prep_ns +=
            ns_between(t_changed_prep_0, t_changed_prep_1);
    }

    std::vector<int> needs_rescan;
    needs_rescan.reserve(active_indices.size());
    for (int idx : active_indices) {
        const int cid = clusters[idx].id;
        if (is_dead_ws[cid]) continue;           // secondary: about to die
        if (is_changed_ws[cid] == 2) continue;   // last-batch main: NN fresh
        if (is_changed_ws[cid] == 1) {
            // earlier-batch main: later batches changed distances, NN stale
            needs_rescan.push_back(idx);
            continue;
        }
        // Bystander: rescan only if NN was invalidated.
        // is_changed_ws[old_nn] catches both ==1 and ==2 (any main).
        const int old_nn = clusters[idx].nn;
        if (old_nn != -1 &&
            (!is_alive_ws[old_nn] || is_dead_ws[old_nn] || is_changed_ws[old_nn])) {
            needs_rescan.push_back(idx);
        }
    }

    if (needs_rescan.empty()) {
        // Clean up (resets both 1 and 2 to 0).
        for (const auto& m : merges) {
            is_changed_ws[m.first] = 0;
            is_dead_ws[m.second] = 0;
        }
        return;
    }

    // Build alive/non-dead scan runs once per iteration.
    // Keeps scan order deterministic while removing per-element alive/dead checks
    // from hot full-rescan paths.
    struct ScanRun {
        int k_start;
        int k_end;
        size_t len;
    };
    std::vector<ScanRun> scan_runs;
    const ScanRun* scan_run_data = nullptr;
    size_t scan_run_count = 0;
    const auto t_scanrun_0 = profile_ops ? std::chrono::high_resolution_clock::now()
                                         : std::chrono::high_resolution_clock::time_point{};
    {
        const char* alive = is_alive_ws.data();
        const char* dead = is_dead_ws.data();
        scan_runs.reserve(active_indices.size() > 0 ? active_indices.size() / 4 : 0);
        int run_start = -1;
        for (int k = 0; k < N; ++k) {
            const bool keep = alive[k] && !dead[k];
            if (keep) {
                if (run_start < 0) run_start = k;
                continue;
            }
            if (run_start >= 0) {
                const int k0 = run_start;
                const int k1 = k - 1;
                scan_runs.push_back(ScanRun{k0, k1, static_cast<size_t>(k1 - k0 + 1)});
                run_start = -1;
            }
        }
        if (run_start >= 0) {
            const int k0 = run_start;
            const int k1 = N - 1;
            scan_runs.push_back(ScanRun{k0, k1, static_cast<size_t>(k1 - k0 + 1)});
        }
        scan_run_data = scan_runs.data();
        scan_run_count = scan_runs.size();
    }
    if (profile_ops) {
        const auto t_scanrun_1 = std::chrono::high_resolution_clock::now();
        g_dense_ops_profile.nn_scan_run_build_ns += ns_between(t_scanrun_0, t_scanrun_1);
    }

    std::atomic<uint64_t> shortlist_attempts_accum{0};
    std::atomic<uint64_t> shortlist_hits_accum{0};
    std::atomic<uint64_t> fullscan_clusters_accum{0};
    std::atomic<long long> shortlist_ns_accum{0};
    std::atomic<long long> fullscan_ns_accum{0};

    auto rescan_range = [&](size_t start, size_t end) {
        const char* dead = is_dead_ws.data();
        const size_t* row_start = dist.row_start.data();
        const SymDistScalar* dist_data = dist.data.data();
        const SymDistScalar inf = std::numeric_limits<SymDistScalar>::infinity();
        uint64_t local_shortlist_attempts = 0;
        uint64_t local_shortlist_hits = 0;
        uint64_t local_fullscan_clusters = 0;
        long long local_shortlist_ns = 0;
        long long local_fullscan_ns = 0;
        auto consider_contiguous_tail = [&](const SymDistScalar* seg,
                                            int k_start,
                                            size_t len,
                                            SymDistScalar& best_val_ref,
                                            int& best_idx_ref) {
#if defined(RACPP_SIMD_TAIL_UPDATE) && RACPP_SIMD_TAIL_UPDATE && \
    defined(__AVX2__) && defined(__FMA__) && \
    defined(RACPP_SYMDIST_USE_FLOAT) && RACPP_SYMDIST_USE_FLOAT
            SymDistScalar seg_best = inf;
            int seg_best_off = -1;
            size_t off = 0;
            for (; off + 7 < len; off += 8) {
                const __m256 v = _mm256_loadu_ps(seg + off);
                const float block_best = hmin_ps256(v);
                if (block_best < seg_best) {
                    seg_best = block_best;
                    alignas(32) float lanes[8];
                    _mm256_store_ps(lanes, v);
                    for (int lane = 0; lane < 8; ++lane) {
                        if (lanes[lane] == block_best) {
                            seg_best_off = static_cast<int>(off) + lane;
                            break;
                        }
                    }
                }
            }
            for (; off < len; ++off) {
                const SymDistScalar v = seg[off];
                if (v < seg_best) {
                    seg_best = v;
                    seg_best_off = static_cast<int>(off);
                }
            }
            if (seg_best_off >= 0 && seg_best < best_val_ref) {
                best_val_ref = seg_best;
                best_idx_ref = k_start + seg_best_off;
            }
#else
            for (size_t off = 0; off < len; ++off) {
                const SymDistScalar v = seg[off];
                if (v < best_val_ref) {
                    best_val_ref = v;
                    best_idx_ref = k_start + static_cast<int>(off);
                }
            }
#endif
        };
        for (size_t i = start; i < end; i++) {
            const int cluster_idx = needs_rescan[i];
            const int cid = clusters[cluster_idx].id;
            SymDistScalar best_val = inf;
            int best_idx = -1;
            bool used_shortcut = false;

            const int old_nn = clusters[cluster_idx].nn;
            const SymDistScalar old_nn_dist =
                static_cast<SymDistScalar>(clusters[cluster_idx].nn_distance);
            const bool nn_invalidated =
                (old_nn != -1) &&
                (dead[old_nn] || is_changed_ws[old_nn]);
            if (use_changed_shortlist &&
                nn_invalidated &&
                old_nn_dist < inf) {
                ++local_shortlist_attempts;
                const auto t_short_0 = profile_ops ? std::chrono::high_resolution_clock::now()
                                                   : std::chrono::high_resolution_clock::time_point{};

                SymDistScalar changed_best = inf;
                int changed_best_idx = -1;
                const size_t cid_base = row_start[static_cast<size_t>(cid)];
                const SymDistScalar* cid_tail = dist_data + cid_base;

                for (size_t r = 0; r < changed_run_count; ++r) {
                    const ChangedMainRun& run = changed_run_data[r];

                    if (run.k_end < cid) {
                        for (int k = run.k_start; k <= run.k_end; ++k) {
                            const SymDistScalar v =
                                dist_data[row_start[static_cast<size_t>(k)] +
                                          static_cast<size_t>(cid - k - 1)];
                            if (v < changed_best ||
                                (v == changed_best &&
                                 (changed_best_idx == -1 || k < changed_best_idx))) {
                                changed_best = v;
                                changed_best_idx = k;
                            }
                        }
                        continue;
                    }

                    if (run.k_start > cid) {
                        const SymDistScalar* seg =
                            cid_tail + static_cast<size_t>(run.k_start - cid - 1);
                        consider_contiguous_tail(
                            seg, run.k_start, run.len, changed_best, changed_best_idx);
                        continue;
                    }

                    // Run intersects cid.
                    for (int k = run.k_start; k < cid; ++k) {
                        const SymDistScalar v =
                            dist_data[row_start[static_cast<size_t>(k)] +
                                      static_cast<size_t>(cid - k - 1)];
                        if (v < changed_best ||
                            (v == changed_best &&
                             (changed_best_idx == -1 || k < changed_best_idx))) {
                            changed_best = v;
                            changed_best_idx = k;
                        }
                    }
                    if (cid < run.k_end) {
                        const int k0 = cid + 1;
                        const size_t len = static_cast<size_t>(run.k_end - cid);
                        const SymDistScalar* seg =
                            cid_tail + static_cast<size_t>(k0 - cid - 1);
                        consider_contiguous_tail(
                            seg, k0, len, changed_best, changed_best_idx);
                    }
                }
                if (profile_ops) {
                    const auto t_short_1 = std::chrono::high_resolution_clock::now();
                    local_shortlist_ns += ns_between(t_short_0, t_short_1);
                }

                // Safe skip of full scan: changed set produced a strictly better NN than
                // previous best distance. Unchanged distances did not change in this round.
                if (changed_best < old_nn_dist) {
                    best_val = changed_best;
                    best_idx = changed_best_idx;
                    used_shortcut = true;
                    ++local_shortlist_hits;
                }
            }

            if (!used_shortcut) {
                if (profile_ops) {
                    ++local_fullscan_clusters;
                }
                const auto t_full_0 = profile_ops ? std::chrono::high_resolution_clock::now()
                                                  : std::chrono::high_resolution_clock::time_point{};
                const size_t cid_base = row_start[static_cast<size_t>(cid)];
                const SymDistScalar* tail = dist_data + cid_base;
                for (size_t r = 0; r < scan_run_count; ++r) {
                    const ScanRun& run = scan_run_data[r];

                    if (run.k_end < cid) {
                        // Entire run is k < cid (scattered triangular reads).
                        for (int k = run.k_start; k <= run.k_end; ++k) {
                            const SymDistScalar v =
                                dist_data[row_start[static_cast<size_t>(k)] +
                                          static_cast<size_t>(cid - k - 1)];
                            if (v < best_val) {
                                best_val = v;
                                best_idx = k;
                            }
                        }
                        continue;
                    }

                    if (run.k_start > cid) {
                        // Entire run is k > cid (contiguous tail in cid row).
                        const int k0 = run.k_start;
                        const SymDistScalar* seg =
                            tail + static_cast<size_t>(k0 - cid - 1);
                        consider_contiguous_tail(seg, k0, run.len, best_val, best_idx);
                        continue;
                    }

                    // Run intersects cid.
                    for (int k = run.k_start; k < cid; ++k) {
                        const SymDistScalar v =
                            dist_data[row_start[static_cast<size_t>(k)] +
                                      static_cast<size_t>(cid - k - 1)];
                        if (v < best_val) {
                            best_val = v;
                            best_idx = k;
                        }
                    }
                    if (cid < run.k_end) {
                        const int k0 = cid + 1;
                        const size_t len = static_cast<size_t>(run.k_end - cid);
                        const SymDistScalar* seg =
                            tail + static_cast<size_t>(k0 - cid - 1);
                        consider_contiguous_tail(seg, k0, len, best_val, best_idx);
                    }
                }
                if (profile_ops) {
                    const auto t_full_1 = std::chrono::high_resolution_clock::now();
                    local_fullscan_ns += ns_between(t_full_0, t_full_1);
                }
            }

            const double best_val_d = static_cast<double>(best_val);
            clusters[cluster_idx].nn = (best_val_d < max_merge_distance) ? best_idx : -1;
            clusters[cluster_idx].nn_distance = best_val_d;
        }
        shortlist_attempts_accum.fetch_add(local_shortlist_attempts, std::memory_order_relaxed);
        shortlist_hits_accum.fetch_add(local_shortlist_hits, std::memory_order_relaxed);
        fullscan_clusters_accum.fetch_add(local_fullscan_clusters, std::memory_order_relaxed);
        if (profile_ops) {
            shortlist_ns_accum.fetch_add(local_shortlist_ns, std::memory_order_relaxed);
            fullscan_ns_accum.fetch_add(local_fullscan_ns, std::memory_order_relaxed);
        }
    };

    const size_t count = needs_rescan.size();
    const size_t no_threads = std::min(requested, count);
    const auto t_rescan_total_0 = profile_ops ? std::chrono::high_resolution_clock::now()
                                              : std::chrono::high_resolution_clock::time_point{};

    if (no_threads <= 1) {
        rescan_range(0, count);
    } else {
        run_parallel_for(requested, count, rescan_range);
    }
    const uint64_t call_shortlist_attempts =
        shortlist_attempts_accum.load(std::memory_order_relaxed);
    const uint64_t call_shortlist_hits =
        shortlist_hits_accum.load(std::memory_order_relaxed);
    if (call_shortlist_attempts > 0) {
        if (call_shortlist_attempts >= SHORTLIST_DISABLE_CALL_MIN_ATTEMPTS &&
            call_shortlist_hits == 0) {
            s_shortlist_disabled.store(1, std::memory_order_relaxed);
        }
        const uint64_t total_attempts =
            s_shortlist_attempts_total.fetch_add(call_shortlist_attempts, std::memory_order_relaxed) +
            call_shortlist_attempts;
        const uint64_t total_hits =
            s_shortlist_hits_total.fetch_add(call_shortlist_hits, std::memory_order_relaxed) +
            call_shortlist_hits;
        if (total_attempts >= SHORTLIST_DISABLE_MIN_ATTEMPTS) {
            const double hit_rate =
                static_cast<double>(total_hits) / static_cast<double>(total_attempts);
            if (hit_rate < SHORTLIST_DISABLE_MIN_HIT_RATE) {
                s_shortlist_disabled.store(1, std::memory_order_relaxed);
            }
        }
    }
    if (profile_ops) {
        const auto t_rescan_total_1 = std::chrono::high_resolution_clock::now();
        g_dense_ops_profile.nn_rescan_total_ns += ns_between(t_rescan_total_0, t_rescan_total_1);
        g_dense_ops_profile.nn_rescan_clusters += static_cast<uint64_t>(count);
        g_dense_ops_profile.nn_shortlist_attempts +=
            shortlist_attempts_accum.load(std::memory_order_relaxed);
        g_dense_ops_profile.nn_shortlist_hits +=
            shortlist_hits_accum.load(std::memory_order_relaxed);
        g_dense_ops_profile.nn_fullscan_clusters +=
            fullscan_clusters_accum.load(std::memory_order_relaxed);
        g_dense_ops_profile.nn_shortlist_eval_ns +=
            shortlist_ns_accum.load(std::memory_order_relaxed);
        g_dense_ops_profile.nn_fullscan_eval_ns +=
            fullscan_ns_accum.load(std::memory_order_relaxed);
    }

    // Clean up (resets both 1 and 2 to 0).
    for (const auto& m : merges) {
        is_changed_ws[m.first] = 0;
        is_dead_ws[m.second] = 0;
    }
}

// Fix #1: Uses active_indices; returns index list instead of pointer list
std::vector<int> get_unique_nn(std::vector<Cluster>& clusters, const std::vector<int>& active_indices, std::vector<int>& nn_count) {
    std::vector<int> unique_nn;
    for (int idx : active_indices) {
        Cluster& cluster = clusters[idx];

        if (cluster.will_merge || (cluster.nn != -1 && clusters[cluster.nn].active && clusters[cluster.nn].will_merge)) {
            if (nn_count[cluster.id] == 0) {
                unique_nn.push_back(idx);
                nn_count[cluster.id]++;
            }
        }
    }

    return unique_nn;
}


void paralell_update_cluster_nn(
    std::vector<Cluster>& clusters,
    const std::vector<int>& active_indices,
    double max_merge_distance,
    size_t no_threads,
    std::vector<int>& nn_count) {

    // Get unique nn indices
    std::vector<int> unique_nn = get_unique_nn(clusters, active_indices, nn_count);
    std::sort(unique_nn.begin(), unique_nn.end());
    const size_t total = unique_nn.size();
    if (total == 0) {
        return;
    }

    const size_t requested_threads = (no_threads > 0) ? no_threads : 1;
    const size_t worker_count = std::min(requested_threads, total);

    if (worker_count <= 1) {
        update_cluster_nn(clusters, unique_nn, max_merge_distance, nn_count);
        return;
    }

    std::vector<std::thread> threads;
    threads.reserve(worker_count);

    const size_t chunk_size = total / worker_count;
    const size_t remainder = total % worker_count;
    size_t start = 0;

    for (size_t t = 0; t < worker_count; t++) {
        const size_t end = start + chunk_size + (t < remainder ? 1 : 0);
        threads.emplace_back([&clusters, &unique_nn, max_merge_distance, &nn_count, start, end]() {
            for (size_t i = start; i < end; i++) {
                const int cluster_idx = unique_nn[i];
                clusters[cluster_idx].update_nn(max_merge_distance);
                nn_count[clusters[cluster_idx].id] = 0;
            }
        });
        start = end;
    }

    for (auto& thread : threads) {
        thread.join();
    }
}

// Fix #1: Uses active_indices instead of iterating full vector with nullptr checks
std::vector<std::pair<int, int> > find_reciprocal_nn(std::vector<Cluster>& clusters, const std::vector<int>& active_indices) {
    std::vector<std::pair<int, int> > reciprocal_nn;

    for (int idx : active_indices) {
        Cluster& cluster = clusters[idx];

        cluster.will_merge = false;

        if (cluster.nn != -1 && clusters[cluster.nn].active) {
            cluster.will_merge = (clusters[cluster.nn].nn == cluster.id);
        }

        if (cluster.will_merge && cluster.id < cluster.nn) {
            reciprocal_nn.push_back(std::make_pair(cluster.id, cluster.nn));
        }
    }

    std::sort(reciprocal_nn.begin(), reciprocal_nn.end());
    return reciprocal_nn;
}

//-----------------------End Updating Nearest Neighbors-----------------------------------

//--------------------------------------RAC Functions--------------------------------------
void RAC_i(
    std::vector<Cluster>& clusters,
    std::vector<int>& active_indices,
    double max_merge_distance,
    const int NO_PROCESSORS,
    std::vector<std::vector<std::pair<int, double>>>& merging_arrays,
    std::vector<int>& sort_neighbor_arr,
    std::vector<std::vector<int>>& update_neighbors_arrays,
    std::vector<int>& nn_count
    ) {

    std::vector<int> active_pos(clusters.size(), -1);
    for (size_t i = 0; i < active_indices.size(); i++) {
        active_pos[active_indices[i]] = static_cast<int>(i);
    }

    std::vector<std::pair<int, int>> merges = find_reciprocal_nn(clusters, active_indices);
    while (merges.size() != 0) {
        update_cluster_dissimilarities(merges, clusters, NO_PROCESSORS, merging_arrays, sort_neighbor_arr, update_neighbors_arrays);

        paralell_update_cluster_nn(clusters, active_indices, max_merge_distance, NO_PROCESSORS, nn_count);

        remove_secondary_clusters(merges, clusters, active_indices, active_pos);

        merges = find_reciprocal_nn(clusters, active_indices);
    }
}

void RAC_i(
    std::vector<Cluster>& clusters,
    std::vector<int>& active_indices,
    double max_merge_distance,
    Eigen::MatrixXd& base_arr,
    const int NO_PROCESSORS) {

    std::vector<int> active_pos(clusters.size(), -1);
    for (size_t i = 0; i < active_indices.size(); i++) {
        active_pos[active_indices[i]] = static_cast<int>(i);
    }

    std::vector<std::pair<int, int>> merges = find_reciprocal_nn(clusters, active_indices);
    while (merges.size() != 0) {
        update_cluster_dissimilarities(merges, clusters, NO_PROCESSORS, base_arr);

        remove_secondary_clusters(merges, clusters, active_indices, active_pos);

        merges = find_reciprocal_nn(clusters, active_indices);
    }
}

void RAC_i(
    std::vector<Cluster>& clusters,
    std::vector<int>& active_indices,
    double max_merge_distance,
    const int NO_PROCESSORS,
    SymDistMatrix& dist,
    std::vector<int>& dsu_parent,
    std::vector<int>& dsu_size
    ) {

    const bool profile_ops = racpp_profile_ops_enabled();
    if (profile_ops) {
        g_dense_ops_profile.reset();
    }

    long total_dissim = 0, total_nn = 0, total_remove = 0, total_find = 0, total_compact = 0;
    int iteration = 0;
    std::vector<SymDistVector> merged_columns_workspace;

    const int orig_N = dist.N;

    std::vector<int> active_pos(clusters.size(), -1);
    for (size_t i = 0; i < active_indices.size(); i++) {
        active_pos[active_indices[i]] = static_cast<int>(i);
    }

    // Persistent workspaces for update_cluster_nn_dist.
    std::vector<char> is_dead_ws(orig_N, 0);
    std::vector<char> is_changed_ws(orig_N, 0);
    std::vector<char> is_alive_ws(orig_N, 0);
    std::vector<char> is_iter_secondary_ws(orig_N, 0);
    for (int idx : active_indices) {
        is_alive_ws[idx] = 1;
    }

    // Compaction state.
    bool did_compact = false;
    int compaction_count = 0;
    std::vector<int> orig_dsu_parent;
    std::vector<int> compact_to_orig;

    std::vector<std::pair<int, int>> merges = find_reciprocal_nn(clusters, active_indices);
    while (!merges.empty()) {
        auto t0 = std::chrono::high_resolution_clock::now();
        update_cluster_dissimilarities(merges, clusters, dist, NO_PROCESSORS,
                                       dsu_parent, dsu_size, merged_columns_workspace,
                                       is_iter_secondary_ws,
                                       max_merge_distance, is_alive_ws);
        auto t1 = std::chrono::high_resolution_clock::now();

        // Mirror merges into original DSU for final label assignment.
        if (did_compact) {
            for (const auto& m : merges) {
                int orig_main = dsu_find(orig_dsu_parent, compact_to_orig[m.first]);
                int orig_sec  = dsu_find(orig_dsu_parent, compact_to_orig[m.second]);
                orig_dsu_parent[orig_sec] = orig_main;
            }
        }

        update_cluster_nn_dist(clusters, active_indices, dist, max_merge_distance,
                               merges, NO_PROCESSORS, is_alive_ws, is_dead_ws, is_changed_ws);
        auto t2 = std::chrono::high_resolution_clock::now();

        remove_secondary_clusters(merges, clusters, active_indices, active_pos);
        for (const auto& merge : merges) {
            is_alive_ws[merge.second] = 0;
        }
        auto t3 = std::chrono::high_resolution_clock::now();

        // Compact repeatedly (with guards) so later iterations scan smaller N.
        // Hysteresis/limits avoid spending too much time in A^2 rebuilds.
        const size_t active_count = active_indices.size();
        const size_t curr_n = static_cast<size_t>(dist.N);
        constexpr size_t MIN_COMPACT_ACTIVE = 64;
        constexpr size_t MIN_COMPACT_N = 8192;
        constexpr size_t MIN_COMPACT_DROP = 4096;
        constexpr int MAX_COMPACTIONS = 2;
        const bool should_compact =
            active_count > MIN_COMPACT_ACTIVE &&
            curr_n >= MIN_COMPACT_N &&
            compaction_count < MAX_COMPACTIONS &&
            (curr_n - active_count) >= MIN_COMPACT_DROP &&
            (active_count * 5 < curr_n * 3);  // active < 60% of current N

        if (should_compact) {

            const int A = static_cast<int>(active_indices.size());
            const int old_N = dist.N;

            std::vector<int> sorted_active = active_indices;
            std::sort(sorted_active.begin(), sorted_active.end());

            std::vector<int> new_compact_to_orig(A);
            std::vector<int> orig_to_compact(dist.N, -1);
            for (int i = 0; i < A; i++) {
                const int old_id = sorted_active[i];
                orig_to_compact[old_id] = i;
                new_compact_to_orig[i] = did_compact
                    ? compact_to_orig[old_id]
                    : old_id;
            }

            // Save original DSU for label assignment on first compaction.
            if (!did_compact) {
                orig_dsu_parent = dsu_parent;
            }

            // Build compact distance matrix (parallel by row).
            SymDistMatrix new_dist(A);
            {
                auto copy_range = [&](size_t start, size_t end) {
                    for (size_t i = start; i < end; i++) {
                        const int ii = static_cast<int>(i);
                        const int oi = sorted_active[ii];
                        for (int j = ii + 1; j < A; j++) {
                            new_dist.data[new_dist.tri_idx(ii, j)] =
                                static_cast<SymDistScalar>(
                                    dist.get(oi, sorted_active[j]));
                        }
                    }
                };

                const size_t req = (NO_PROCESSORS > 0) ? static_cast<size_t>(NO_PROCESSORS) : 1;
                const size_t nt = std::min(req, static_cast<size_t>(A));
                if (nt <= 1) {
                    copy_range(0, A);
                } else {
                    run_parallel_for(req, static_cast<size_t>(A), copy_range);
                }
            }
            dist = std::move(new_dist);

            // Rebuild clusters with compact IDs.
            std::vector<Cluster> new_clusters;
            new_clusters.reserve(A);
            for (int i = 0; i < A; i++) {
                new_clusters.emplace_back(i);
                Cluster& oc = clusters[sorted_active[i]];
                const bool has_mapped_nn = (oc.nn >= 0 && orig_to_compact[oc.nn] >= 0);
                new_clusters[i].nn = has_mapped_nn ? orig_to_compact[oc.nn] : -1;
                new_clusters[i].nn_distance =
                    has_mapped_nn ? oc.nn_distance : std::numeric_limits<double>::infinity();
            }
            clusters = std::move(new_clusters);

            // Rebuild active state.
            active_indices.resize(A);
            std::iota(active_indices.begin(), active_indices.end(), 0);
            active_pos.assign(A, 0);
            std::iota(active_pos.begin(), active_pos.end(), 0);

            // Rebuild compact DSU.
            std::vector<int> new_dsu_size(A);
            for (int i = 0; i < A; i++) {
                new_dsu_size[i] = dsu_size[sorted_active[i]];
            }
            dsu_parent.resize(A);
            std::iota(dsu_parent.begin(), dsu_parent.end(), 0);
            dsu_size = std::move(new_dsu_size);

            compact_to_orig = std::move(new_compact_to_orig);
            // Resize workspaces.
            merged_columns_workspace.clear();
            is_alive_ws.assign(A, 1);
            is_dead_ws.assign(A, 0);
            is_changed_ws.assign(A, 0);
            is_iter_secondary_ws.assign(A, 0);

            did_compact = true;
            compaction_count++;

            auto tc = std::chrono::high_resolution_clock::now();
            total_compact += std::chrono::duration_cast<std::chrono::milliseconds>(tc - t3).count();
            std::cerr << "Compacted: " << old_N << " -> " << A
                      << " ("
                      << std::chrono::duration_cast<std::chrono::milliseconds>(tc - t3).count()
                      << "ms)" << std::endl;
        }

        auto t4_start = std::chrono::high_resolution_clock::now();
        merges = find_reciprocal_nn(clusters, active_indices);
        auto t4 = std::chrono::high_resolution_clock::now();

        total_dissim += std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();
        total_nn += std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count();
        total_remove += std::chrono::duration_cast<std::chrono::milliseconds>(t3 - t2).count();
        total_find += std::chrono::duration_cast<std::chrono::milliseconds>(t4 - t4_start).count();
        iteration++;
    }

    // Restore original DSU for label assignment.
    if (did_compact) {
        dsu_parent = std::move(orig_dsu_parent);
    }

    std::cerr << "RAC_i iterations: " << iteration
              << " | dissim: " << total_dissim << "ms"
              << " | nn_dist: " << total_nn << "ms"
              << " | remove: " << total_remove << "ms"
              << " | find_rnn: " << total_find << "ms"
              << " | compact: " << total_compact << "ms" << std::endl;
    if (profile_ops) {
        std::cerr << "RAC profile (dissim): cross=" << ns_to_ms(g_dense_ops_profile.dissim_cross_ns) << "ms"
                  << " | cand_build=" << ns_to_ms(g_dense_ops_profile.dissim_candidate_build_ns) << "ms"
                  << " | compute=" << ns_to_ms(g_dense_ops_profile.dissim_compute_ns) << "ms"
                  << " | writeback=" << ns_to_ms(g_dense_ops_profile.dissim_writeback_ns) << "ms"
                  << " | last_batch_nn=" << ns_to_ms(g_dense_ops_profile.dissim_last_batch_nn_ns) << "ms"
                  << std::endl;
        std::cerr << "RAC profile (nn_dist): shortlist_prep="
                  << ns_to_ms(g_dense_ops_profile.nn_changed_shortlist_prep_ns) << "ms"
                  << " | scan_run_build=" << ns_to_ms(g_dense_ops_profile.nn_scan_run_build_ns) << "ms"
                  << " | rescan_total=" << ns_to_ms(g_dense_ops_profile.nn_rescan_total_ns) << "ms"
                  << " | shortlist_eval=" << ns_to_ms(g_dense_ops_profile.nn_shortlist_eval_ns) << "ms"
                  << " | fullscan_eval=" << ns_to_ms(g_dense_ops_profile.nn_fullscan_eval_ns) << "ms"
                  << " | rescans=" << g_dense_ops_profile.nn_rescan_clusters
                  << " | shortlist_attempts=" << g_dense_ops_profile.nn_shortlist_attempts
                  << " | shortlist_hits=" << g_dense_ops_profile.nn_shortlist_hits
                  << " | fullscan_clusters=" << g_dense_ops_profile.nn_fullscan_clusters
                  << std::endl;
    }
}

template <typename Scalar>
static std::vector<int> RAC_impl_no_connectivity(
    Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>& base_arr,
    double max_merge_distance,
    int no_processors,
    const std::string& distance_metric) {

    const int NO_PROCESSORS = (no_processors != 0) ? no_processors : getProcessorCount();
    const int N = static_cast<int>(base_arr.cols());
    Eigen::setNbThreads(NO_PROCESSORS);

    std::vector<Cluster> clusters;
    clusters.reserve(N);
    std::vector<int> active_indices;
    active_indices.reserve(N);
    for (int i = 0; i < N; ++i) {
        clusters.emplace_back(i);
        active_indices.push_back(i);
    }

    auto start = std::chrono::high_resolution_clock::now();
    SymDistMatrix dist = calculate_initial_dissimilarities_dense(base_arr, clusters, max_merge_distance, distance_metric);
    auto end = std::chrono::high_resolution_clock::now();
    std::cout << "Initial Dissimilarities: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms" << std::endl;

    // Free base_arr — no longer needed after initial dissimilarities are computed.
    base_arr.resize(0, 0);

    // DSU for O(1) merges and O(N) label assignment
    std::vector<int> dsu_parent(N);
    std::iota(dsu_parent.begin(), dsu_parent.end(), 0);
    std::vector<int> dsu_size(N, 1);

    RAC_i(clusters, active_indices, max_merge_distance, NO_PROCESSORS, dist, dsu_parent, dsu_size);

    // Label assignment via DSU: O(N α(N)) ≈ O(N)
    std::vector<int> cluster_labels(N);
    for (int i = 0; i < N; i++) {
        cluster_labels[i] = dsu_find(dsu_parent, i);
    }
    return cluster_labels;
}

// Internal implementation: expects D×N column-major data (already transposed + normalized).
static std::vector<int> RAC_impl(
    Eigen::MatrixXd& base_arr,
    double max_merge_distance,
    Eigen::SparseMatrix<bool>* connectivity,
    int batch_size,
    int no_processors,
    std::string distance_metric) {

    if (connectivity == nullptr) {
        return RAC_impl_no_connectivity(base_arr, max_merge_distance, no_processors, distance_metric);
    }

    const int NO_PROCESSORS = (no_processors != 0) ? no_processors : getProcessorCount();
    const int N = static_cast<int>(base_arr.cols());
    const int BATCHSIZE = (batch_size != 0) ? batch_size : std::max(1, N / 10);

    Eigen::setNbThreads(NO_PROCESSORS);

    std::vector<Cluster> clusters;
    clusters.reserve(N);
    std::vector<int> active_indices;
    active_indices.reserve(N);
    for (int i = 0; i < N; ++i) {
        clusters.emplace_back(i);
        active_indices.push_back(i);
    }

    auto start = std::chrono::high_resolution_clock::now();

    std::vector<std::vector<std::pair<int, double>>> merging_arrays(NO_PROCESSORS, std::vector<std::pair<int, double>>(clusters.size()));
    std::vector<int> sort_neighbor_arr(clusters.size(), -1);
    std::vector<std::vector<int>> update_neighbors_arrays(NO_PROCESSORS, std::vector<int>(clusters.size()));
    std::vector<int> nn_count(clusters.size(), 0);

    calculate_initial_dissimilarities(base_arr, clusters, *connectivity, max_merge_distance, BATCHSIZE, distance_metric);
    auto end = std::chrono::high_resolution_clock::now();
    std::cout << "Initial Dissimilarities: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms" << std::endl;

    // Free base_arr — no longer needed after initial dissimilarities are computed.
    base_arr.resize(0, 0);

    RAC_i(clusters, active_indices, max_merge_distance, NO_PROCESSORS, merging_arrays, sort_neighbor_arr, update_neighbors_arrays, nn_count);

    // Label assignment via cluster indices (connectivity path still uses indices)
    std::vector<int> cluster_labels(N);
    for (int idx : active_indices) {
        const Cluster& cluster = clusters[idx];
        for (int index : cluster.indices) {
            cluster_labels[index] = cluster.id;
        }
    }
    return cluster_labels;
}

// Public C++ API: accepts N×D input (rows=points, cols=dimensions).
std::vector<int> RAC(
    const Eigen::MatrixXd& base_arr_in,
    double max_merge_distance,
    Eigen::SparseMatrix<bool>* connectivity,
    int batch_size = 0,
    int no_processors = 0,
    std::string distance_metric = "euclidean") {

    // const int NO_PROCESSORS = resolve_processor_count(no_processors);
    // Eigen::setNbThreads(NO_PROCESSORS);

    // Transpose (+ normalize for cosine) into D×N working copy.
    const auto t_pre_start = std::chrono::high_resolution_clock::now();
    Eigen::MatrixXd base_arr;
    if (distance_metric == "cosine") {
        base_arr = base_arr_in.transpose().colwise().normalized();
    } else {
        base_arr = base_arr_in.transpose();
    }
    const auto t_pre_end = std::chrono::high_resolution_clock::now();
    std::cout << "Preprocessing: "
              << std::chrono::duration_cast<std::chrono::milliseconds>(t_pre_end - t_pre_start).count()
              << "ms" << std::endl;

    return RAC_impl(base_arr, max_merge_distance, connectivity, batch_size, no_processors, distance_metric);
}
//--------------------------------------End RAC Functions--------------------------------------


//------------------------PYBIND INTERFACE----------------------------------

#if !RACPP_BUILDING_LIB_ONLY
//Wrapper for RAC, convert return vector to a numpy array
py::array RAC_py(
    py::array base_arr_np,
    double max_merge_distance,
    py::object connectivity = py::none(),
    int batch_size = 0,
    int no_processors = 0,
    std::string distance_metric = "cosine") {

    auto buf = base_arr_np.request();
    if (buf.ndim != 2) {
        throw py::value_error("base_arr must be a 2D numpy.ndarray with shape (N, D)");
    }
    if (buf.shape[0] <= 0 || buf.shape[1] <= 0) {
        throw py::value_error("base_arr must have positive shape (N > 0, D > 0).");
    }
    if ((base_arr_np.flags() & py::array::c_style) == 0) {
        throw py::value_error("base_arr must be C-contiguous. Use np.ascontiguousarray(base_arr).");
    }
    if (batch_size < 0) {
        throw py::value_error("batch_size must be >= 0.");
    }
    if (no_processors < 0) {
        throw py::value_error("no_processors must be >= 0.");
    }
    if (distance_metric != "cosine" && distance_metric != "euclidean") {
        throw py::value_error("distance_metric must be either 'cosine' or 'euclidean'.");
    }

    const int N = static_cast<int>(buf.shape[0]);
    const int D = static_cast<int>(buf.shape[1]);
    // const int NO_PROCESSORS = resolve_processor_count(no_processors);
    // Eigen::setNbThreads(NO_PROCESSORS);

    const std::string format = buf.format;
    const bool is_float64 = (format == py::format_descriptor<double>::format());
    const bool is_float32 = (format == py::format_descriptor<float>::format());
    if (!is_float64 && !is_float32) {
        throw py::value_error("base_arr dtype must be float32 or float64.");
    }

    std::shared_ptr<Eigen::SparseMatrix<bool>> sparse_connectivity = nullptr;
    if (!connectivity.is_none()) {
        sparse_connectivity = std::make_shared<Eigen::SparseMatrix<bool>>(connectivity.cast<Eigen::SparseMatrix<bool>>());
        if (sparse_connectivity->rows() != N || sparse_connectivity->cols() != N) {
            throw py::value_error("connectivity must have shape (N, N) matching base_arr rows.");
        }
    }

    if (is_float64) {
        const auto t_pre_start = std::chrono::high_resolution_clock::now();
        Eigen::Map<const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>> base_transposed(
            static_cast<const double*>(buf.ptr), D, N);

        Eigen::MatrixXd base_arr;
        if (distance_metric == "cosine") {
            base_arr = base_transposed.colwise().normalized();
        } else {
            base_arr = base_transposed;
        }
        const auto t_pre_end = std::chrono::high_resolution_clock::now();
        std::cout << "Preprocessing: "
                  << std::chrono::duration_cast<std::chrono::milliseconds>(t_pre_end - t_pre_start).count()
                  << "ms" << std::endl;

        std::vector<int> cluster_labels = RAC_impl(
            base_arr,
            max_merge_distance,
            sparse_connectivity.get(),
            batch_size,
            no_processors,
            distance_metric);
        return py::cast(cluster_labels);
    }

    // Float32 input: keep full no-connectivity flow in float32.
    const auto t_pre_start = std::chrono::high_resolution_clock::now();
    Eigen::Map<const Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic>> base_transposed(
        static_cast<const float*>(buf.ptr), D, N);

    if (connectivity.is_none()) {
        Eigen::MatrixXf base_arr;
        if (distance_metric == "cosine") {
            base_arr = base_transposed.colwise().normalized();
        } else {
            base_arr = base_transposed;
        }
        const auto t_pre_end = std::chrono::high_resolution_clock::now();
        std::cout << "Preprocessing: "
                  << std::chrono::duration_cast<std::chrono::milliseconds>(t_pre_end - t_pre_start).count()
                  << "ms" << std::endl;

        std::vector<int> cluster_labels = RAC_impl_no_connectivity(
            base_arr,
            max_merge_distance,
            no_processors,
            distance_metric);
        return py::cast(cluster_labels);
    }

    // Connectivity flow is currently double-precision; keep float32 support via single cast.
    Eigen::MatrixXd base_arr;
    if (distance_metric == "cosine") {
        base_arr = base_transposed.cast<double>().colwise().normalized();
    } else {
        base_arr = base_transposed.cast<double>();
    }
    const auto t_pre_end = std::chrono::high_resolution_clock::now();
    std::cout << "Preprocessing: "
              << std::chrono::duration_cast<std::chrono::milliseconds>(t_pre_end - t_pre_start).count()
              << "ms" << std::endl;

    std::vector<int> cluster_labels = RAC_impl(
        base_arr,
        max_merge_distance,
        sparse_connectivity.get(),
        batch_size,
        no_processors,
        distance_metric);
    return py::cast(cluster_labels);
}

//Wrapper for pairwise euclidean distance
py::array _pairwise_euclidean_distance_py(
    Eigen::MatrixXd base_arr,
    Eigen::MatrixXd query_arr) {

    base_arr = base_arr.transpose().eval();
    query_arr = query_arr.transpose().eval();

    Eigen::MatrixXd distance_matrix = pairwise_euclidean(base_arr, query_arr);
    py::array distance_matrix_arr = py::cast(distance_matrix);
    return distance_matrix_arr;
}

//Wrapper for pairwise cosine distance
py::array _pairwise_cosine_distance_py(
    Eigen::MatrixXd base_arr,
    Eigen::MatrixXd query_arr) {

    base_arr = base_arr.transpose().colwise().normalized().eval();
    query_arr = query_arr.transpose().colwise().normalized().eval();

    Eigen::MatrixXd distance_matrix = pairwise_cosine(base_arr, query_arr);
    py::array distance_matrix_arr = py::cast(distance_matrix);
    return distance_matrix_arr;
}


void simple_pybind_io_test() {
    std::cout << std::endl;
    std::cout << "This is a simple pybind I/O Test." << std::endl;
    std::cout << std::endl;
}

PYBIND11_MODULE(_racplusplus, m){
    m.doc() = R"doc(
        RACplusplus is a C++ optimized python package for performing
        reciprocal agglomerative clustering.

        Authors: Porter Hunley, Daniel Frees
        2023
    )doc";

    m.def("rac", &RAC_py,
        py::arg("base_arr"),
        py::arg("max_merge_distance"),
        py::arg("connectivity") = py::none(),
        py::arg("batch_size") = 0,
        py::arg("no_processors") = 0,
        py::arg("distance_metric") = "euclidean",
        R"fdoc(
        Run RAC algorithm on a provided array of points.

        Params:
        [base_arr] -        Actual data points array to be clustered. Each row is a point, with each column
                            representing the points value for a particular feature/dimension.
        [max_merge_distance] - Hyperparameter, maximum distance allowed for two clusters to merge with one another.
        [connectivity] -    Optional: Connectivity matrix indicating whether points can be considered as neighbors.
                            Value of 1 at index i,j indicates point i and j are connected, 0 indicates disconnected.
                            Default: No connectivity matrix, use pairwise cosine to calculate distances.
        [batch_size] -      Optional hyperparameter, batch size for calculating initial dissimilarities
                            with a connectivity matrix.
                            Default: Defaults to the number of points in base_arr / 10 if 0 passed or no value passed.
        [no_processors] -   Hyperparameter, number of processors to use during computation.
                            Defaults to the number of processors found on your machine if 0 passed
                            or no value passed.
        [distance_metric] - Optional: Distance metric to use for calculating distances between points.
                            Default: Euclidean distance.

        Output:
        Returns a numpy array of the group # each point in base_arr was assigned to.
    )fdoc");

    m.def("_pairwise_euclidean_distance", &_pairwise_euclidean_distance_py, R"fdoc(
        Calculate pairwise euclidean distance

        Params:
            [base_arr] -        Actual data points array to be clustered. Each row is a point, with each column
                                representing the points value for a particular feature/dimension.
            [query_arr] -       Actual data points array to be clustered. Each row is a point, with each column
        Output:
        Returns a numpy distance array
    )fdoc");

    m.def("_pairwise_cosine_distance", &_pairwise_cosine_distance_py, R"fdoc(
        Calculate pairwise cosine distance

        Params:
            [base_arr] -        Actual data points array to be clustered. Each row is a point, with each column
                                representing the points value for a particular feature/dimension.
            [query_arr] -       Actual data points array to be clustered. Each row is a point, with each column
        Output:
        Returns a numpy distance array
    )fdoc");

    m.def("test_rac", &racplusplus_cli_test, R"fdoc(
        Testing function to run and time RAC's run in C++.
    )fdoc");

    m.def("simple_pybind_io_test", &simple_pybind_io_test, R"fdoc(
        Simple test function to see if pybind works, and can print text in python.
    )fdoc");

    m.attr("__version__") = "0.9";
#if defined(RACPP_SYMDIST_USE_FLOAT) && RACPP_SYMDIST_USE_FLOAT
    m.attr("__symdist_storage__") = "float32";
#else
    m.attr("__symdist_storage__") = "float64";
#endif
}
//------------------------END PYBIND INTERFACE----------------------------------
#endif
