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
// #define EIGEN_DONT_PARALLELIZE
#include "Eigen/Dense"
#include "Eigen/Sparse"
#include <random>
#include <numeric>
#include <cmath>

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
    }


void Cluster::update_nn(double max_merge_distance) {
    if (neighbor_distances.size() == 0) {
        nn = -1;
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

    if (min < max_merge_distance) {
        this->nn = nn;
    } else {
        this->nn = -1;
    }
}

void Cluster::update_nn(const SymDistMatrix& dist, double max_merge_distance) {
    auto [min_val, min_idx] = dist.min_in_col(this->id);

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

void remove_secondary_clusters(std::vector<std::pair<int, int> >& merges, std::vector<Cluster>& clusters, std::vector<int>& active_indices) {
    for (const auto& merge : merges) {
        clusters[merge.second].active = false;
    }
    active_indices.erase(
        std::remove_if(active_indices.begin(), active_indices.end(),
            [&clusters](int id) { return !clusters[id].active; }),
        active_indices.end());
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

//Averaged dissimilarity across two matrices (wrapper for pairwise distance calc + avging)
// Fix #4: take by const reference instead of by value
double calculate_weighted_dissimilarity(const Eigen::MatrixXd& points_a, const Eigen::MatrixXd& points_b) {
    Eigen::MatrixXd dissimilarity_matrix = pairwise_cosine(points_a, points_b);

    return static_cast<double>(dissimilarity_matrix.mean());
}

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

// Fix #2: Two-phase full-matrix merge to avoid data races on shared distance_arr.
// Phase 1 computes updated columns in parallel (read-only), phase 2 writes back serially.
// Merges are processed in batches to cap peak memory — each batch pre-allocates at most
// MERGE_BATCH columns (≈480 MB for 60k) instead of one per merge (could be 4+ GB).
// Cross-batch interactions are handled by the updated dist; intra-batch by the patch.
void update_cluster_dissimilarities(
    std::vector<std::pair<int, int> >& merges,
    std::vector<Cluster>& clusters,
    SymDistMatrix& dist,
    const int NO_PROCESSORS,
    std::vector<int>& dsu_parent,
    std::vector<int>& dsu_size) {

    if (merges.empty()) {
        return;
    }

    const size_t merge_count = merges.size();
    const int N = dist.N;
    const size_t MERGE_BATCH = 1024;

    // Pre-allocate reusable column buffers — capped at MERGE_BATCH.
    const size_t max_batch = std::min(merge_count, MERGE_BATCH);
    std::vector<Eigen::VectorXd> merged_columns(max_batch);
    for (size_t i = 0; i < max_batch; i++) {
        merged_columns[i].resize(N);
    }

    for (size_t batch_start = 0; batch_start < merge_count; batch_start += MERGE_BATCH) {
        const size_t batch_end = std::min(batch_start + MERGE_BATCH, merge_count);
        const size_t batch_size = batch_end - batch_start;

        // Prep merge metadata for this batch.
        // Sizes come from dsu_size — correct because merges are disjoint, so
        // this batch's clusters haven't been touched by earlier batches' unions.
        std::vector<int> merge_main_ids(batch_size);
        std::vector<int> merge_secondary_ids(batch_size);
        std::vector<double> merge_main_sizes(batch_size);
        std::vector<double> merge_secondary_sizes(batch_size);
        std::vector<double> merge_inv_sizes(batch_size);

        for (size_t i = 0; i < batch_size; i++) {
            const size_t global_i = batch_start + i;
            merge_main_ids[i] = merges[global_i].first;
            merge_secondary_ids[i] = merges[global_i].second;

            merge_main_sizes[i] = static_cast<double>(dsu_size[merge_main_ids[i]]);
            merge_secondary_sizes[i] = static_cast<double>(dsu_size[merge_secondary_ids[i]]);
            merge_inv_sizes[i] = 1.0 / (merge_main_sizes[i] + merge_secondary_sizes[i]);
        }

        // Parallel compute for this batch.
        auto compute_range = [&](size_t start, size_t end) {
            const double inf = std::numeric_limits<double>::infinity();
            Eigen::VectorXd main_col(N);
            Eigen::VectorXd sec_col(N);

            for (size_t i = start; i < end; i++) {
                const int main_id = merge_main_ids[i];
                const int secondary_id = merge_secondary_ids[i];
                const double main_size = merge_main_sizes[i];
                const double secondary_size = merge_secondary_sizes[i];
                const double inv_ab = merge_inv_sizes[i];

                dist.get_col_into(main_id, main_col);
                dist.get_col_into(secondary_id, sec_col);
                merged_columns[i].noalias() = (main_size * main_col + secondary_size * sec_col) * inv_ab;

                // Patch: only for other merges within THIS batch.
                // Cross-batch merges are already reflected in dist from earlier write-backs.
                for (size_t j = 0; j < batch_size; j++) {
                    if (merge_main_ids[j] == main_id || merge_secondary_ids[j] == main_id) continue;

                    const int merge_main = merge_main_ids[j];
                    const int merge_secondary = merge_secondary_ids[j];
                    const double inv_cd = merge_inv_sizes[j];

                    const double dist_main_to_cd =
                        (merge_main_sizes[j] * dist.get(merge_main, main_id) +
                         merge_secondary_sizes[j] * dist.get(merge_secondary, main_id)) * inv_cd;
                    const double dist_secondary_to_cd =
                        (merge_main_sizes[j] * dist.get(merge_main, secondary_id) +
                         merge_secondary_sizes[j] * dist.get(merge_secondary, secondary_id)) * inv_cd;

                    const double dist_ab_to_cd = (main_size * dist_main_to_cd + secondary_size * dist_secondary_to_cd) * inv_ab;
                    merged_columns[i][merge_main] = dist_ab_to_cd;
                    merged_columns[i][merge_secondary] = dist_ab_to_cd;
                }

                merged_columns[i][main_id] = inf;
                merged_columns[i][secondary_id] = inf;
            }
        };

        size_t requested_threads = (NO_PROCESSORS > 0) ? static_cast<size_t>(NO_PROCESSORS) : 1;
        size_t no_threads = std::min(requested_threads, batch_size);

        if (no_threads <= 1) {
            compute_range(0, batch_size);
        } else {
            std::vector<std::thread> threads;
            threads.reserve(no_threads);
            size_t chunk_size = batch_size / no_threads;
            size_t remainder = batch_size % no_threads;
            size_t start = 0;
            for (size_t t = 0; t < no_threads; t++) {
                size_t end = start + chunk_size + (t < remainder ? 1 : 0);
                threads.emplace_back(compute_range, start, end);
                start = end;
            }
            for (auto& th : threads) th.join();
        }

        // Serial write-back: update dist columns first, then fill infinity for secondaries.
        // Order matters: set_col must complete for all merges before fill_infinity,
        // otherwise fill_infinity(B) could be overwritten by a later set_col(C) writing to dist[C][B].
        for (size_t i = 0; i < batch_size; i++) {
            dist.set_col(merge_main_ids[i], merged_columns[i]);
            dsu_union(dsu_parent, dsu_size, merge_main_ids[i], merge_secondary_ids[i]);
        }
        for (size_t i = 0; i < batch_size; i++) {
            dist.fill_infinity(merge_secondary_ids[i]);
        }
    }
}

SymDistMatrix calculate_initial_dissimilarities(
    Eigen::MatrixXd& base_arr,
    std::vector<Cluster>& clusters,
    double max_merge_distance,
    std::string distance_metric) {

    const int N = static_cast<int>(clusters.size());
    const int D = static_cast<int>(base_arr.rows());
    const int TILE = 4096;
    const bool is_cosine = (distance_metric == "cosine");

    SymDistMatrix dist(N);

    // Per-cluster NN tracking.
    std::vector<double> nn_best(N, std::numeric_limits<double>::infinity());
    std::vector<int> nn_idx(N, -1);

    // Pre-compute squared norms for euclidean distance.
    // Use same scalar type as tiles so broadcasts don't require conversion.
#if defined(RACPP_SYMDIST_USE_FLOAT) && RACPP_SYMDIST_USE_FLOAT
    Eigen::VectorXf sq_norms;
    if (!is_cosine) {
        sq_norms = base_arr.colwise().squaredNorm().cast<float>();
    }
#else
    Eigen::VectorXd sq_norms;
    if (!is_cosine) {
        sq_norms = base_arr.colwise().squaredNorm();
    }
#endif

    // Tiled pairwise distance computation.
    // Only compute upper-triangle tile pairs (j_start >= i_start).
    for (int i_start = 0; i_start < N; i_start += TILE) {
        const int i_end = std::min(i_start + TILE, N);
        const int tile_i = i_end - i_start;

        // Block view into base_arr columns [i_start, i_end) — no copy.
        auto Bi = base_arr.block(0, i_start, D, tile_i);
#if defined(RACPP_SYMDIST_USE_FLOAT) && RACPP_SYMDIST_USE_FLOAT
        // Pre-cast Bi to float once per outer iteration (enables SGEMM).
        Eigen::MatrixXf Bi_f = Bi.cast<float>();
#endif

        for (int j_start = i_start; j_start < N; j_start += TILE) {
            const int j_end = std::min(j_start + TILE, N);
            const int tile_j = j_end - j_start;

            auto Bj = base_arr.block(0, j_start, D, tile_j);

            // GEMM: tile_i × tile_j. Uses Eigen's internal BLAS threading.
#if defined(RACPP_SYMDIST_USE_FLOAT) && RACPP_SYMDIST_USE_FLOAT
            Eigen::MatrixXf tile = Bi_f.transpose() * Bj.cast<float>();
#else
            Eigen::MatrixXd tile = Bi.transpose() * Bj;
#endif

            // Apply distance transform in-place.
            if (is_cosine) {
                tile = (-tile).array() + 1.0;
            } else {
                tile *= -2.0;
                for (int r = 0; r < tile_i; r++) {
                    tile.row(r).array() += sq_norms[i_start + r];
                }
                for (int c = 0; c < tile_j; c++) {
                    tile.col(c).array() += sq_norms[j_start + c];
                }
                tile = tile.array().max(0.0).sqrt();
            }

            // Store to SymDistMatrix + track NNs.
            for (int r = 0; r < tile_i; r++) {
                const int i_global = i_start + r;
                for (int c = 0; c < tile_j; c++) {
                    const int j_global = j_start + c;
                    if (i_global >= j_global) continue;

                    const double val = tile(r, c);
                    dist.data[dist.tri_idx(i_global, j_global)] = val;

                    if (val < nn_best[i_global]) {
                        nn_best[i_global] = val;
                        nn_idx[i_global] = j_global;
                    }
                    if (val < nn_best[j_global]) {
                        nn_best[j_global] = val;
                        nn_idx[j_global] = i_global;
                    }
                }
            }
        }
    }

    // Set cluster NNs.
    for (int k = 0; k < N; k++) {
        clusters[k].nn = (nn_best[k] < max_merge_distance) ? nn_idx[k] : -1;
    }

    return dist;
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

    for (int batchStart = 0; batchStart < clustersSize; batchStart += batch_size) {
        int batchEnd = std::min(batchStart + batch_size, clustersSize);

        for (int i = batchStart; i < batchEnd; ++i) {
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
        Eigen::MatrixXd full_main = base_arr(Eigen::all, main_cluster.indices);
        Eigen::MatrixXd full_other = base_arr(Eigen::all, other_cluster_idxs);
        double dist = pairwise_cosine(full_main, full_other).mean();

        return dist;
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

std::vector<std::vector<std::pair<int, int> > > chunk_merges(std::vector<std::pair<int, int> >& merges, size_t no_threads) {
    std::vector<std::vector<std::pair<int, int> > > merge_chunks(no_threads);

    size_t chunk_size = merges.size() / no_threads;
    size_t remainder = merges.size() % no_threads;

    size_t start = 0, end = 0;
    for (size_t i = 0; i < no_threads; i++) {
        end = start + chunk_size;
        if (i < remainder) { // distribute the remainder among the first "remainder" chunks
            end++;
        }

        // Create chunks by using the range constructor of std::vector
        if (end <= merges.size()) {
            merge_chunks[i] = std::vector<std::pair<int, int> >(merges.begin() + start, merges.begin() + end);
        }
        start = end;
    }

    return merge_chunks;
}

void parallel_merge_clusters(
    std::vector<std::pair<int, int> >& merges,
    std::vector<Cluster>& clusters,
    size_t no_threads,
    std::vector<std::vector<std::pair<int, double>>>& merging_arrays) {

    std::vector<std::thread> threads;

    std::vector<std::vector<std::pair<int, int>>> merge_chunks;
    merge_chunks = chunk_merges(merges, no_threads);

    for (size_t i=0; i<no_threads; i++) {
        std::thread merge_thread = std::thread(
            merge_clusters_symmetric,
            std::ref(merge_chunks[i]),
            std::ref(clusters),
            std::ref(merging_arrays[i]));

        threads.push_back(std::move(merge_thread));
    }

    for (size_t i=0; i<no_threads; i++) {
        threads[i].join();
    }
}

void parallel_merge_clusters(
    std::vector<std::pair<int, int> >& merges,
    std::vector<Cluster>& clusters,
    size_t no_threads,
    std::vector<std::vector<int>>& merging_arrays,
    Eigen::MatrixXd& base_arr) {

    std::vector<std::thread> threads;

    std::vector<std::vector<std::pair<int, int>>> merge_chunks;
    merge_chunks = chunk_merges(merges, no_threads);

    for (size_t i=0; i<no_threads; i++) {
        std::thread merge_thread = std::thread(
            merge_clusters_compute,
            std::ref(merge_chunks[i]),
            std::ref(clusters),
            std::ref(merging_arrays[i]),
            std::ref(base_arr));

        threads.push_back(std::move(merge_thread));
    }

    for (size_t i=0; i<no_threads; i++) {
        threads[i].join();
    }
}
//-----------------------End Merging Functions-----------------------------------

//-----------------------Updating Nearest Neighbors-----------------------------------

void update_cluster_neighbors(
    std::pair<int, std::vector<std::pair<int, double> > >& update_chunk,
    std::vector<Cluster>& clusters,
    std::vector<int>& update_neighbors) {
    Cluster& other_cluster = clusters[update_chunk.first];

    int no_updates = update_chunk.second.size();
    int no_neighbors = other_cluster.neighbor_distances.size();

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

    std::vector<std::thread> threads;
    std::vector<std::vector<std::pair<int, std::vector<std::pair<int, double>>>>> update_chunks(no_threads);

    size_t chunk_size = updates.size() / no_threads;
    size_t remainder = updates.size() % no_threads;

    size_t start = 0, end = 0;
    for (size_t i = 0; i < no_threads; i++) {
        end = start + chunk_size;
        if (i < remainder) { // distribute the remainder among the first "remainder" chunks
            end++;
        }

        if (end <= updates.size()) {
            update_chunks[i] = std::vector<std::pair<int, std::vector<std::pair<int, double> > > >(updates.begin() + start, updates.begin() + end);
        }
        start = end;
    }

    for (size_t i=0; i<no_threads; i++) {
        std::thread update_thread = std::thread(
            update_cluster_neighbors_p,
            std::ref(update_chunks[i]),
            std::ref(clusters),
            std::ref(neighbor_sort_arr),
            std::ref(update_neighbors_arrays[i]));

        threads.push_back(std::move(update_thread));
    }

    for (size_t i=0; i<no_threads; i++) {
        threads[i].join();
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
    const int NO_PROCESSORS) {

    // Collect indices that actually need updating.
    std::vector<int> needs_update;
    needs_update.reserve(active_indices.size());
    for (int idx : active_indices) {
        Cluster& cluster = clusters[idx];
        if (cluster.will_merge || (cluster.nn != -1 && clusters[cluster.nn].active && clusters[cluster.nn].will_merge)) {
            needs_update.push_back(idx);
        }
    }

    if (needs_update.empty()) return;

    auto update_range = [&](size_t start, size_t end) {
        for (size_t i = start; i < end; i++) {
            clusters[needs_update[i]].update_nn(dist, max_merge_distance);
        }
    };

    size_t count = needs_update.size();
    size_t requested = (NO_PROCESSORS > 0) ? static_cast<size_t>(NO_PROCESSORS) : 1;
    size_t no_threads = std::min(requested, count);

    if (no_threads <= 1) {
        update_range(0, count);
    } else {
        std::vector<std::thread> threads;
        threads.reserve(no_threads);

        size_t chunk = count / no_threads;
        size_t remainder = count % no_threads;
        size_t start = 0;

        for (size_t t = 0; t < no_threads; t++) {
            size_t end = start + chunk + (t < remainder ? 1 : 0);
            threads.emplace_back(update_range, start, end);
            start = end;
        }

        for (auto& th : threads) {
            th.join();
        }
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

    std::vector<std::thread> threads;
    std::vector<std::vector<int>> index_chunks(no_threads);

    size_t chunk_size = unique_nn.size() / no_threads;
    size_t remainder = unique_nn.size() % no_threads;

    size_t start = 0, end = 0;
    for (size_t i = 0; i < no_threads; i++) {
        end = start + chunk_size;
        if (i < remainder) {
            end++;
        }

        if (end <= unique_nn.size()) {
            index_chunks[i] = std::vector<int>(unique_nn.begin() + start, unique_nn.begin() + end);
        }
        start = end;
    }

    for (size_t i=0; i<no_threads; i++) {
        std::thread update_thread = std::thread(
            update_cluster_nn,
            std::ref(clusters),
            std::ref(index_chunks[i]),
            max_merge_distance,
            std::ref(nn_count));

        threads.push_back(std::move(update_thread));
    }

    for (size_t i=0; i<no_threads; i++) {
        threads[i].join();
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

    std::vector<std::pair<int, int>> merges = find_reciprocal_nn(clusters, active_indices);
    while (merges.size() != 0) {
        update_cluster_dissimilarities(merges, clusters, NO_PROCESSORS, merging_arrays, sort_neighbor_arr, update_neighbors_arrays);

        paralell_update_cluster_nn(clusters, active_indices, max_merge_distance, NO_PROCESSORS, nn_count);

        remove_secondary_clusters(merges, clusters, active_indices);

        merges = find_reciprocal_nn(clusters, active_indices);
    }
}

void RAC_i(
    std::vector<Cluster>& clusters,
    std::vector<int>& active_indices,
    double max_merge_distance,
    Eigen::MatrixXd& base_arr,
    const int NO_PROCESSORS) {

    std::vector<std::pair<int, int>> merges = find_reciprocal_nn(clusters, active_indices);
    while (merges.size() != 0) {
        update_cluster_dissimilarities(merges, clusters, NO_PROCESSORS, base_arr);

        remove_secondary_clusters(merges, clusters, active_indices);

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

    long total_dissim = 0, total_nn = 0, total_remove = 0, total_find = 0;
    int iteration = 0;

    std::vector<std::pair<int, int>> merges = find_reciprocal_nn(clusters, active_indices);
    while (merges.size() != 0) {
        auto t0 = std::chrono::high_resolution_clock::now();
        update_cluster_dissimilarities(merges, clusters, dist, NO_PROCESSORS, dsu_parent, dsu_size);
        auto t1 = std::chrono::high_resolution_clock::now();

        update_cluster_nn_dist(clusters, active_indices, dist, max_merge_distance, NO_PROCESSORS);
        auto t2 = std::chrono::high_resolution_clock::now();

        remove_secondary_clusters(merges, clusters, active_indices);
        auto t3 = std::chrono::high_resolution_clock::now();

        merges = find_reciprocal_nn(clusters, active_indices);
        auto t4 = std::chrono::high_resolution_clock::now();

        total_dissim += std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();
        total_nn += std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count();
        total_remove += std::chrono::duration_cast<std::chrono::milliseconds>(t3 - t2).count();
        total_find += std::chrono::duration_cast<std::chrono::milliseconds>(t4 - t3).count();
        iteration++;
    }

    std::cerr << "RAC_i iterations: " << iteration
              << " | dissim: " << total_dissim << "ms"
              << " | nn_dist: " << total_nn << "ms"
              << " | remove: " << total_remove << "ms"
              << " | find_rnn: " << total_find << "ms" << std::endl;
}

// Internal implementation: expects D×N column-major data (already transposed + normalized).
static std::vector<int> RAC_impl(
    Eigen::MatrixXd& base_arr,
    double max_merge_distance,
    Eigen::SparseMatrix<bool>* connectivity,
    int batch_size,
    int no_processors,
    std::string distance_metric) {

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

    if (connectivity == nullptr) {
        SymDistMatrix dist = calculate_initial_dissimilarities(base_arr, clusters, max_merge_distance, distance_metric);
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
    } else {
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
}

// Public C++ API: accepts N×D input (rows=points, cols=dimensions).
std::vector<int> RAC(
    const Eigen::MatrixXd& base_arr_in,
    double max_merge_distance,
    Eigen::SparseMatrix<bool>* connectivity,
    int batch_size = 0,
    int no_processors = 0,
    std::string distance_metric = "euclidean") {

    // Transpose (+ normalize for cosine) into D×N working copy.
    Eigen::MatrixXd base_arr;
    if (distance_metric == "cosine") {
        base_arr = base_arr_in.transpose().colwise().normalized();
    } else {
        base_arr = base_arr_in.transpose();
    }

    return RAC_impl(base_arr, max_merge_distance, connectivity, batch_size, no_processors, distance_metric);
}
//--------------------------------------End RAC Functions--------------------------------------


//------------------------PYBIND INTERFACE----------------------------------

#if !RACPP_BUILDING_LIB_ONLY
//Wrapper for RAC, convert return vector to a numpy array
py::array RAC_py(
    py::array_t<double, py::array::c_style | py::array::forcecast> base_arr_np,
    double max_merge_distance,
    py::object connectivity = py::none(),
    int batch_size = 0,
    int no_processors = 0,
    std::string distance_metric = "euclidean") {

    auto buf = base_arr_np.request();
    const int N = static_cast<int>(buf.shape[0]);
    const int D = static_cast<int>(buf.shape[1]);

    // Zero-copy transpose: C-contiguous (N,D) numpy == column-major (D,N) Eigen.
    Eigen::Map<const Eigen::MatrixXd> base_transposed(
        static_cast<const double*>(buf.ptr), D, N);

    // One allocation: D×N working copy (normalized if cosine).
    Eigen::MatrixXd base_arr;
    if (distance_metric == "cosine") {
        base_arr = base_transposed.colwise().normalized();
    } else {
        base_arr = base_transposed;
    }

    std::shared_ptr<Eigen::SparseMatrix<bool>> sparse_connectivity = nullptr;
    if (!connectivity.is_none()) {
        sparse_connectivity = std::make_shared<Eigen::SparseMatrix<bool>>(connectivity.cast<Eigen::SparseMatrix<bool>>());
    }

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

    m.def("rac", &RAC_py, R"fdoc(
        Run RAC algorithm on a provided array of points.

        Params:
        [base_arr] -        Actual data points array to be clustered. Each row is a point, with each column
                            representing the points value for a particular feature/dimension.
        [max_merge_distance] - Hyperparameter, maximum distance allowed for two clusters to merge with one another.
        [batch_size] -      Optional hyperparameter, batch size for calculating initial dissimilarities
                            with a connectivity matrix.
                            Default: Defaults to the number of points in base_arr / 10 if 0 passed or no value passed.
        [connectivity] -    Optional: Connectivity matrix indicating whether points can be considered as neighbors.
                            Value of 1 at index i,j indicates point i and j are connected, 0 indicates disconnected.
                            Default: No connectivity matrix, use pairwise cosine to calculate distances.
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
