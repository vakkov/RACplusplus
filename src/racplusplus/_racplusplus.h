#ifndef RACPP_BUILDING_LIB_ONLY
#define RACPP_BUILDING_LIB_ONLY 0
#endif

#include <array>
#include <tuple>
#include <vector>
#include <set>
#include <limits>
#include <algorithm>
#include "Eigen/Dense"
#include "Eigen/Sparse"

#ifndef GLOBAL_TIMING_VARS_H
#define GLOBAL_TIMING_VARS_H

// Store update neighbor times
extern std::vector<long> UPDATE_NEIGHBOR_DURATIONS;
// Store update NN times
extern std::vector<long> UPDATE_NN_DURATIONS;
// Store the durations of each call to cosine
extern std::vector<long> COSINE_DURATIONS;
extern std::vector<long> INDICES_DURATIONS;
extern std::vector<long> MERGE_DURATIONS;
extern std::vector<long> MISC_MERGE_DURATIONS;
extern std::vector<long> INITIAL_NEIGHBOR_DURATIONS;
extern std::vector<long> HASH_DURATIONS;
extern std::vector<double> UPDATE_PERCENTAGES;

#endif // GLOBAL_TIMING_VARS_H

#ifndef CLUSTER_H
#define CLUSTER_H

class SymDistMatrix;  // forward declaration

class Cluster {
public:
    int id;
    bool will_merge;
    bool active;
    int nn;
    std::vector<std::pair<int, double>> neighbor_distances;
    std::vector<int> neighbors;
    std::vector<int> indices;
    std::vector<std::pair<int, double>> dissimilarities; // sorted by .first for binary search
    std::vector<std::tuple<int, int, double> > neighbors_needing_updates;

    Cluster(int id);

    void update_nn(double max_merge_distance);
    void update_nn(const SymDistMatrix& dist, double max_merge_distance);
};

#endif //CLUSTER_H

#ifndef SYMDISTMATRIX_H
#define SYMDISTMATRIX_H

#if defined(RACPP_SYMDIST_USE_FLOAT) && RACPP_SYMDIST_USE_FLOAT
using SymDistScalar = float;
#else
using SymDistScalar = double;
#endif

class SymDistMatrix {
public:
    int N;
    std::vector<SymDistScalar> data;

    explicit SymDistMatrix(int n)
        : N(n), data(static_cast<size_t>(n) * (n - 1) / 2,
               std::numeric_limits<SymDistScalar>::infinity()) {}

    inline size_t tri_idx(int i, int j) const {
        return static_cast<size_t>(i) * N
             - static_cast<size_t>(i) * (i + 1) / 2
             + j - i - 1;
    }

    inline double get(int i, int j) const {
        if (i == j) return std::numeric_limits<double>::infinity();
        if (i > j) std::swap(i, j);
        return static_cast<double>(data[tri_idx(i, j)]);
    }

    inline void set(int i, int j, double val) {
        if (i == j) return;
        if (i > j) std::swap(i, j);
        data[tri_idx(i, j)] = static_cast<SymDistScalar>(val);
    }

    Eigen::VectorXd get_col(int col_id) const {
        Eigen::VectorXd col(N);
        get_col_into(col_id, col);
        return col;
    }

    // Fill an existing VectorXd (must already be size N) — no heap allocation.
    void get_col_into(int col_id, Eigen::VectorXd& col) const {
        // k < col_id: scattered access with decreasing stride
        for (int k = 0; k < col_id; ++k) {
            col[k] = static_cast<double>(data[tri_idx(k, col_id)]);
        }
        col[col_id] = std::numeric_limits<double>::infinity();
        // k > col_id: contiguous access starting at tri_idx(col_id, col_id+1)
        if (col_id + 1 < N) {
            size_t base = tri_idx(col_id, col_id + 1);
            for (int k = col_id + 1; k < N; ++k) {
                col[k] = static_cast<double>(data[base + (k - col_id - 1)]);
            }
        }
    }

    void set_col(int col_id, const Eigen::VectorXd& col) {
        // k < col_id: scattered access
        for (int k = 0; k < col_id; ++k) {
            data[tri_idx(k, col_id)] = static_cast<SymDistScalar>(col[k]);
        }
        // k > col_id: contiguous access
        if (col_id + 1 < N) {
            size_t base = tri_idx(col_id, col_id + 1);
            for (int k = col_id + 1; k < N; ++k) {
                data[base + (k - col_id - 1)] = static_cast<SymDistScalar>(col[k]);
            }
        }
    }

    // Find minimum value in a "column" without allocating a VectorXd.
    // Returns (min_value, min_index).
    // Uses split-loop to avoid branch per element and exploit contiguous access for k > col_id.
    std::pair<double, int> min_in_col(int col_id) const {
        double best_val = std::numeric_limits<double>::infinity();
        int best_idx = -1;

        // k < col_id: scattered access
        for (int k = 0; k < col_id; ++k) {
            double v = static_cast<double>(data[tri_idx(k, col_id)]);
            if (v < best_val) {
                best_val = v;
                best_idx = k;
            }
        }

        // k > col_id: contiguous access
        if (col_id + 1 < N) {
            size_t base = tri_idx(col_id, col_id + 1);
            for (int k = col_id + 1; k < N; ++k) {
                double v = static_cast<double>(data[base + (k - col_id - 1)]);
                if (v < best_val) {
                    best_val = v;
                    best_idx = k;
                }
            }
        }

        return {best_val, best_idx};
    }

    void fill_infinity(int cluster_id) {
        const SymDistScalar inf = std::numeric_limits<SymDistScalar>::infinity();
        // k < cluster_id: scattered access
        for (int k = 0; k < cluster_id; ++k) {
            data[tri_idx(k, cluster_id)] = inf;
        }
        // k > cluster_id: contiguous access
        if (cluster_id + 1 < N) {
            size_t base = tri_idx(cluster_id, cluster_id + 1);
            size_t count = static_cast<size_t>(N - cluster_id - 1);
            std::fill_n(data.data() + base, count, inf);
        }
    }
};

#endif // SYMDISTMATRIX_H

//--------------------Helpers------------------------------------
//Function to optimize to # of processors
size_t getProcessorCount();

// Function to generate a matrix filled with random numbers.
Eigen::MatrixXd generateRandomMatrix(int rows, int cols, int seed);

double get_arr_value(Eigen::MatrixXd& arr, int i, int j);
void set_arr_value(Eigen::MatrixXd& arr, int i, int j, double value);

void remove_secondary_clusters(std::vector<std::pair<int, int> >& merges, std::vector<Cluster>& clusters, std::vector<int>& active_indices);
//--------------------End Helpers------------------------------------


//-----------------------Distance Calculations-------------------------
//Calculate pairwise cosines between two matrices
Eigen::MatrixXd pairwise_cosine(const Eigen::MatrixXd& A, const Eigen::MatrixXd& B);

//Calculate pairwise euclidean between two matrices
Eigen::MatrixXd pairwise_euclidean(const Eigen::MatrixXd& array_a, const Eigen::MatrixXd& array_b);

//Averaged dissimilarity across two matrices (wrapper for pairwise distance calc + avging)
double calculate_weighted_dissimilarity(const Eigen::MatrixXd& points_a, const Eigen::MatrixXd& points_b);

void update_cluster_dissimilarities(
    std::vector<std::pair<int, int> >& merges,
    std::vector<Cluster>& clusters,
    const int NO_PROCESSORS,
    Eigen::MatrixXd& base_arr);

void update_cluster_dissimilarities(
    std::vector<std::pair<int, int> >& merges,
    std::vector<Cluster>& clusters,
    const int NO_PROCESSORS,
    std::vector<std::vector<std::pair<int, double>>>& merging_arrays,
    std::vector<int>& sort_neighbor_arr,
    std::vector<std::vector<int>>& update_neighbors_arrays);

void update_cluster_dissimilarities(
    std::vector<std::pair<int, int> >& merges,
    std::vector<Cluster>& clusters,
    SymDistMatrix& dist,
    const int NO_PROCESSORS);

SymDistMatrix calculate_initial_dissimilarities(
    Eigen::MatrixXd& base_arr,
    std::vector<Cluster>& clusters,
    double max_merge_distance,
    std::string distance_metric);

void calculate_initial_dissimilarities(
    Eigen::MatrixXd& base_arr,
    std::vector<Cluster>& clusters,
    Eigen::SparseMatrix<bool>& connectivity,
    double max_merge_distance,
    int batch_size,
    std::string distance_metric);

//-----------------------End Distance Calculations-------------------------

//-----------------------Merging Functions-----------------------------------
void merge_cluster_compute_linkage(
    std::pair<int, int>& merge,
    std::vector<Cluster>& clusters,
    std::vector<int>& merging_array,
    Eigen::MatrixXd& base_arr);

void merge_cluster_symmetric_linkage(
    std::pair<int, int>& merge,
    std::vector<Cluster>& clusters,
    std::vector<std::pair<int, double>>& merging_array);

void merge_clusters_compute(
    std::vector<std::pair<int, int> >& merges,
    std::vector<Cluster>& clusters,
    std::vector<int>& merging_array,
    Eigen::MatrixXd& base_arr);

void merge_clusters_symmetric(
    std::vector<std::pair<int, int> >& merges,
    std::vector<Cluster>& clusters,
    std::vector<std::pair<int, double>>& merging_array);

void parallel_merge_clusters(
    std::vector<std::pair<int, int> >& merges,
    std::vector<Cluster>& clusters,
    size_t no_threads,
    std::vector<std::vector<int>>& merging_arrays,
    Eigen::MatrixXd& base_arr);

void parallel_merge_clusters(
    std::vector<std::pair<int, int> >& merges,
    std::vector<Cluster>& clusters,
    size_t no_threads,
    std::vector<std::vector<std::pair<int, double>>>& merging_arrays);
//-----------------------End Merging Functions-----------------------------------

//-----------------------Updating Nearest Neighbors-----------------------------------

void update_cluster_neighbors(
    std::pair<int, std::vector<std::pair<int, double> > >& update_chunk,
    std::vector<Cluster>& clusters,
    std::vector<int>& update_neighbors);

void update_cluster_neighbors(
    SymDistMatrix& dist,
    std::vector<std::pair<int, int>> merges);

void update_cluster_neighbors_p(
    std::vector<std::pair<int, std::vector<std::pair<int, double> > > >& updates,
    std::vector<Cluster>& clusters,
    std::vector<int>& neighbor_sort_arr,
    std::vector<int>& update_neighbors);

void parallel_update_clusters(
    std::vector<std::pair<int, std::vector<std::pair<int, double> > > >& updates,
    std::vector<Cluster>& clusters,
    std::vector<std::vector<int>>& update_neighbors_arrays,
    std::vector<int>& neighbor_sort_arr,
    size_t no_threads);

void update_cluster_nn(
    std::vector<Cluster>& clusters,
    const std::vector<int>& indices_to_update,
    double max_merge_distance,
    std::vector<int>& nn_count);

void update_cluster_nn_dist(
    std::vector<Cluster>& clusters,
    const std::vector<int>& active_indices,
    const SymDistMatrix& dist,
    double max_merge_distance,
    const int NO_PROCESSORS);

std::vector<std::pair<int, int> > find_reciprocal_nn(std::vector<Cluster>& clusters, const std::vector<int>& active_indices);
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
    std::vector<int>& nn_count);

void RAC_i(
    std::vector<Cluster>& clusters,
    std::vector<int>& active_indices,
    double max_merge_distance,
    Eigen::MatrixXd& base_arr,
    const int NO_PROCESSORS);

void RAC_i(
    std::vector<Cluster>& clusters,
    std::vector<int>& active_indices,
    double max_merge_distance,
    const int NO_PROCESSORS,
    SymDistMatrix& dist);

std::vector<int> RAC(
    Eigen::MatrixXd& base_arr,
    double max_merge_distance,
    Eigen::SparseMatrix<bool>* connectivity,
    int batch_size,
    int no_processors,
    std::string distance_metric);

#if !RACPP_BUILDING_LIB_ONLY
py::array RAC_py(
    Eigen::MatrixXd base_arr,
    double max_merge_distance,
    py::object connectivity,
    int batch_size,
    int no_processors,
    std::string distance_metric);

py::array _pairwise_euclidean_distance_py(
    Eigen::MatrixXd base_arr,
    Eigen::MatrixXd query_arr);

py::array _pairwise_cosine_distance_py(
    Eigen::MatrixXd base_arr,
    Eigen::MatrixXd query_arr);
#endif
//--------------------------------------End RAC Functions--------------------------------------
