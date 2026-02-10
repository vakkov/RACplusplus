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

void Cluster::update_nn(Eigen::MatrixXd& distance_arr, double max_merge_distance) {
    Eigen::MatrixXd::Index minRow;
    distance_arr.col(this->id).minCoeff(&minRow);

    double min = distance_arr(minRow, this->id);
    int nn = static_cast<int>(minRow);

    if (min < max_merge_distance) {
        this->nn = nn;
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
void update_cluster_dissimilarities(
    std::vector<std::pair<int, int> >& merges,
    std::vector<Cluster>& clusters,
    Eigen::MatrixXd& distance_arr,
    const int NO_PROCESSORS) {

    if (merges.empty()) {
        return;
    }

    // Two-phase merge update:
    //   1) Parallel, read-only compute of updated columns (based on a snapshot of cluster sizes + distance_arr)
    //   2) Serial write-back into distance_arr and cluster index vectors
    //
    // This avoids data races from concurrent writes to shared state while restoring parallel speed
    // for the full distance-matrix (no-connectivity) hot path.

    const size_t merge_count = merges.size();

    std::vector<int> cluster_sizes(clusters.size());
    for (size_t i = 0; i < clusters.size(); i++) {
        cluster_sizes[i] = static_cast<int>(clusters[i].indices.size());
    }

    std::vector<int> merge_main_ids(merge_count);
    std::vector<int> merge_secondary_ids(merge_count);
    std::vector<double> merge_main_sizes(merge_count);
    std::vector<double> merge_secondary_sizes(merge_count);
    std::vector<double> merge_inv_sizes(merge_count);

    for (size_t i = 0; i < merge_count; i++) {
        merge_main_ids[i] = merges[i].first;
        merge_secondary_ids[i] = merges[i].second;

        merge_main_sizes[i] = static_cast<double>(cluster_sizes[merge_main_ids[i]]);
        merge_secondary_sizes[i] = static_cast<double>(cluster_sizes[merge_secondary_ids[i]]);

        merge_inv_sizes[i] = 1.0 / (merge_main_sizes[i] + merge_secondary_sizes[i]);
    }

    std::vector<Eigen::VectorXd> merged_columns(merge_count);

    auto compute_range = [&](size_t start, size_t end) {
        const double inf = std::numeric_limits<double>::infinity();

        for (size_t i = start; i < end; i++) {
            const int main_id = merge_main_ids[i];
            const int secondary_id = merge_secondary_ids[i];

            const double main_size = merge_main_sizes[i];
            const double secondary_size = merge_secondary_sizes[i];
            const double inv_ab = merge_inv_sizes[i];

            // Base: average-linkage merge for (main, secondary) against all clusters.
            Eigen::VectorXd avg_col = (main_size * distance_arr.col(main_id) + secondary_size * distance_arr.col(secondary_id)) * inv_ab;

            // Patch entries for other merges so distances reflect their merged clusters (not their pre-merge constituents).
            for (size_t j = 0; j < merge_count; j++) {
                if (merge_main_ids[j] == main_id || merge_secondary_ids[j] == main_id) {
                    continue;
                }

                const int merge_main = merge_main_ids[j];
                const int merge_secondary = merge_secondary_ids[j];
                const double inv_cd = merge_inv_sizes[j];

                const double dist_main_to_cd =
                    (merge_main_sizes[j] * distance_arr(merge_main, main_id) +
                     merge_secondary_sizes[j] * distance_arr(merge_secondary, main_id)) *
                    inv_cd;
                const double dist_secondary_to_cd =
                    (merge_main_sizes[j] * distance_arr(merge_main, secondary_id) +
                     merge_secondary_sizes[j] * distance_arr(merge_secondary, secondary_id)) *
                    inv_cd;

                const double dist_ab_to_cd = (main_size * dist_main_to_cd + secondary_size * dist_secondary_to_cd) * inv_ab;
                avg_col[merge_main] = dist_ab_to_cd;
                avg_col[merge_secondary] = dist_ab_to_cd;
            }

            avg_col[main_id] = inf;
            avg_col[secondary_id] = inf;

            merged_columns[i] = std::move(avg_col);
        }
    };

    size_t requested_threads = (NO_PROCESSORS > 0) ? static_cast<size_t>(NO_PROCESSORS) : static_cast<size_t>(1);
    size_t no_threads = std::min(requested_threads, merge_count);

    if (no_threads <= 1) {
        compute_range(0, merge_count);
    } else {
        std::vector<std::thread> threads;
        threads.reserve(no_threads);

        size_t chunk_size = merge_count / no_threads;
        size_t remainder = merge_count % no_threads;

        size_t start = 0;
        for (size_t t = 0; t < no_threads; t++) {
            size_t end = start + chunk_size + (t < remainder ? 1 : 0);
            threads.emplace_back(compute_range, start, end);
            start = end;
        }

        for (auto& th : threads) {
            th.join();
        }
    }

    // Serial write-back.
    for (size_t i = 0; i < merge_count; i++) {
        const int main_id = merge_main_ids[i];
        const int secondary_id = merge_secondary_ids[i];

        distance_arr.col(main_id) = merged_columns[i];

        Cluster& main_cluster = clusters[main_id];
        Cluster& secondary_cluster = clusters[secondary_id];

        main_cluster.indices.reserve(main_cluster.indices.size() + secondary_cluster.indices.size());
        main_cluster.indices.insert(main_cluster.indices.end(), secondary_cluster.indices.begin(), secondary_cluster.indices.end());
        secondary_cluster.indices.clear();
    }

    update_cluster_neighbors(distance_arr, merges);
}

Eigen::MatrixXd calculate_initial_dissimilarities(
    Eigen::MatrixXd& base_arr,
    std::vector<Cluster>& clusters,
    double max_merge_distance,
    std::string distance_metric) {

    Eigen::MatrixXd distance_mat;
    if (distance_metric == "cosine") {
        distance_mat = pairwise_cosine(base_arr, base_arr);
    } else {
        distance_mat = pairwise_euclidean(base_arr, base_arr);
    }

    size_t clusterSize = clusters.size();
    for (size_t i=0; i<clusterSize; i++) {
        double min = std::numeric_limits<double>::infinity();
        int nn = -1;

        Cluster& currentCluster = clusters[i];

        for (size_t j=0; j<clusterSize; j++) {
            if (i == j) {
                distance_mat(i, j) = std::numeric_limits<double>::infinity();
                continue;
            }

            double distance = distance_mat(i, j);
            if (distance < max_merge_distance) {
                currentCluster.neighbors.push_back(j);

                if (distance < min) {
                    min = distance;
                    nn = j;
                }
            }
        }

        currentCluster.nn = nn;
    }

    return distance_mat;
}

void calculate_initial_dissimilarities(
    Eigen::MatrixXd& base_arr,
    std::vector<Cluster>& clusters,
    Eigen::SparseMatrix<bool>& connectivity,
    double max_merge_distance,
    int batch_size,
    std::string distance_metric) {

    int clustersSize = static_cast<int>(clusters.size());
    for (int batchStart = 0; batchStart < clustersSize; batchStart += batch_size) {
        int batchEnd = std::min(batchStart + batch_size, clustersSize);
        Eigen::MatrixXd batch = base_arr.block(0, clusters[batchStart].indices[0], base_arr.rows(), clusters[batchEnd - 1].indices[0] - clusters[batchStart].indices[0] + 1);

        Eigen::MatrixXd distance_mat;
        if (distance_metric == "cosine") {
            distance_mat = pairwise_cosine(base_arr, batch);
        } else {
            distance_mat = pairwise_euclidean(base_arr, batch);
        }

        for (int i = batchStart; i < batchEnd; ++i) {
            Cluster& cluster = clusters[i];
            Eigen::VectorXd distance_vec = distance_mat.col(i - batchStart);

            std::vector<std::pair<int, double>> neighbors;

            int nearest_neighbor = -1;
            double min = std::numeric_limits<double>::infinity();

            Eigen::SparseVector<bool> cluster_column = connectivity.innerVector(i);
            for (Eigen::SparseVector<bool>::InnerIterator it(cluster_column); it; ++it) {
                int j = it.index();
                bool value = it.value();

                if (j != i && value) {
                    neighbors.push_back(std::make_pair(j, distance_vec[j]));

                    if (distance_vec[j] < min && distance_vec[j] < max_merge_distance) {
                        min = distance_vec[j];
                        nearest_neighbor = j;
                    }
                }
            }

            cluster.neighbor_distances = neighbors;
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

// Merges with the full distance array
void merge_cluster_full(
    std::pair<int, int>& merge,
    std::vector<std::pair<int, int>>& merges,
    std::vector<Cluster>& clusters,
    Eigen::MatrixXd& distance_arr) {

    Cluster& main_cluster = clusters[merge.first];
    Cluster& secondary_cluster = clusters[merge.second];

    // Get main and secondary columns from distance array
    Eigen::VectorXd main_col = distance_arr.col(main_cluster.id);
    Eigen::VectorXd secondary_col = distance_arr.col(secondary_cluster.id);


    // Loop through merges and change main_col vals
    for (auto& m : merges) {
        if (m.first == main_cluster.id || m.second == main_cluster.id) {
            continue;
        }

        int merge_main = m.first;
        int merge_secondary = m.second;

        int merge_main_size = clusters[merge_main].indices.size();
        int merge_secondary_size = clusters[merge_secondary].indices.size();

        main_col[merge_main] = (merge_main_size * main_col[merge_main] + merge_secondary_size * main_col[merge_secondary]) / (merge_main_size + merge_secondary_size);
        main_col[merge_secondary] = main_col[merge_main];

        secondary_col[merge_main] = (merge_main_size * secondary_col[merge_main] + merge_secondary_size * secondary_col[merge_secondary]) / (merge_main_size + merge_secondary_size);
        secondary_col[merge_secondary] = secondary_col[merge_main];
    }

    int main_size = main_cluster.indices.size();
    int secondary_size = secondary_cluster.indices.size();

    // average main_col and secondary_col
    Eigen::VectorXd avg_col = (main_size * main_col + secondary_size * secondary_col) / (main_size + secondary_size);
    avg_col[secondary_cluster.id] = std::numeric_limits<double>::infinity();

    // Swap main and secondary columns for avg col
    distance_arr.col(main_cluster.id) = avg_col;

    main_cluster.indices.insert(main_cluster.indices.end(), secondary_cluster.indices.begin(), secondary_cluster.indices.end());
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
    Eigen::MatrixXd& distance_arr,
    std::vector<std::pair<int, int>> merges
) {
    // copy merges columns to rows in distance_arr
    for (size_t i=0; i<merges.size(); i++) {
        int cluster_id = merges[i].first;

        distance_arr.row(cluster_id) = distance_arr.col(cluster_id);
    }

    for (size_t i=0; i<merges.size(); i++) {
        int cluster_id = merges[i].second;

        distance_arr.col(cluster_id) = Eigen::VectorXd::Constant(distance_arr.cols(), std::numeric_limits<double>::infinity());
        distance_arr.row(cluster_id) = distance_arr.col(cluster_id);
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
void update_cluster_nn_dist(
    std::vector<Cluster>& clusters,
    const std::vector<int>& active_indices,
    Eigen::MatrixXd& distance_arr,
    double max_merge_distance) {
    for (int idx : active_indices) {
        Cluster& cluster = clusters[idx];

        if (cluster.will_merge || (cluster.nn != -1 && clusters[cluster.nn].active && clusters[cluster.nn].will_merge)) {
            cluster.update_nn(distance_arr, max_merge_distance);
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
    Eigen::MatrixXd& distance_arr
    ) {

    std::vector<std::pair<int, int>> merges = find_reciprocal_nn(clusters, active_indices);
    while (merges.size() != 0) {
        update_cluster_dissimilarities(merges, clusters, distance_arr, NO_PROCESSORS);

        update_cluster_nn_dist(clusters, active_indices, distance_arr, max_merge_distance);

        remove_secondary_clusters(merges, clusters, active_indices);

        merges = find_reciprocal_nn(clusters, active_indices);
    }
}

// Fix #1: Single entry point with vector<Cluster> (contiguous memory), active_indices, no new/delete
std::vector<int> RAC(
    Eigen::MatrixXd& base_arr,
    double max_merge_distance,
    Eigen::SparseMatrix<bool>* connectivity,
    int batch_size = 0,
    int no_processors = 0,
    std::string distance_metric = "euclidean") {

    //Processor Count defaults to the number on the machine if not provided or -1 passed
    const int NO_PROCESSORS = (no_processors != 0) ? no_processors : getProcessorCount();

    //Collect number of points in base_arr for space allocation
    const int NO_POINTS = base_arr.rows();

    //Batch Size defaults to NO_POINTS / 10 if not provided or -1 passed
    const int BATCHSIZE = (batch_size != 0) ? batch_size : NO_POINTS / 10;

    if (distance_metric == "cosine") {
        base_arr = base_arr.transpose().colwise().normalized().eval();
    } else {
        base_arr = base_arr.transpose().eval();
    }

    Eigen::setNbThreads(NO_PROCESSORS);

    // Fix #1: Contiguous vector<Cluster> instead of scattered heap allocations
    std::vector<Cluster> clusters;
    clusters.reserve(base_arr.cols());
    std::vector<int> active_indices;
    active_indices.reserve(base_arr.cols());
    for (long i = 0; i < base_arr.cols(); ++i) {
        clusters.emplace_back(static_cast<int>(i));
        active_indices.push_back(static_cast<int>(i));
    }

    auto start = std::chrono::high_resolution_clock::now();

    if (connectivity == nullptr) {
        Eigen::MatrixXd distance_arr = calculate_initial_dissimilarities(base_arr, clusters, max_merge_distance, distance_metric);
        auto end = std::chrono::high_resolution_clock::now();
        std::cout << "Initial Dissimilarities: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms" << std::endl;

        RAC_i(clusters, active_indices, max_merge_distance, NO_PROCESSORS, distance_arr);
    } else {
        std::vector<std::vector<std::pair<int, double>>> merging_arrays(NO_PROCESSORS, std::vector<std::pair<int, double>>(clusters.size()));
        std::vector<int> sort_neighbor_arr(clusters.size(), -1);
        std::vector<std::vector<int>> update_neighbors_arrays(NO_PROCESSORS, std::vector<int>(clusters.size()));
        std::vector<int> nn_count(clusters.size(), 0);

        calculate_initial_dissimilarities(base_arr, clusters, *connectivity, max_merge_distance, BATCHSIZE, distance_metric);
        auto end = std::chrono::high_resolution_clock::now();
        std::cout << "Initial Dissimilarities: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms" << std::endl;

        RAC_i(clusters, active_indices, max_merge_distance, NO_PROCESSORS, merging_arrays, sort_neighbor_arr, update_neighbors_arrays, nn_count);
    }

    // Direct-index label assignment: O(N) instead of O(N log N) sort
    std::vector<int> cluster_labels(base_arr.cols());
    for (int idx : active_indices) {
        const Cluster& cluster = clusters[idx];
        for (int index : cluster.indices) {
            cluster_labels[index] = cluster.id;
        }
    }

    // No manual delete needed — vector<Cluster> cleans up automatically

    return cluster_labels;
}
//--------------------------------------End RAC Functions--------------------------------------


//------------------------PYBIND INTERFACE----------------------------------

#if !RACPP_BUILDING_LIB_ONLY
//Wrapper for RAC, convert return vector to a numpy array
py::array RAC_py(
    Eigen::MatrixXd base_arr,
    double max_merge_distance,
    py::object connectivity = py::none(),
    int batch_size = 0,
    int no_processors = 0,
    std::string distance_metric = "euclidean") {

    std::shared_ptr<Eigen::SparseMatrix<bool>> sparse_connectivity = nullptr;

    if (!connectivity.is_none()) {
        sparse_connectivity = std::make_shared<Eigen::SparseMatrix<bool>>(connectivity.cast<Eigen::SparseMatrix<bool>>());
    }

    std::vector<int> cluster_labels = RAC(
        base_arr,
        max_merge_distance,
        sparse_connectivity.get(),
        batch_size,
        no_processors,
        distance_metric);

    py::array cluster_labels_arr =  py::cast(cluster_labels);
    return cluster_labels_arr;
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
}
//------------------------END PYBIND INTERFACE----------------------------------
#endif
