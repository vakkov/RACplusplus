#ifndef RACPP_BUILDING_LIB_ONLY
#define RACPP_BUILDING_LIB_ONLY 0
#endif

#include <array>
#include <tuple>
#include <vector>
#include <set>
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
    void update_nn(Eigen::MatrixXd& distance_arr, double max_merge_distance);
};

#endif //CLUSTER_H

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
    Eigen::MatrixXd& distance_arr,
    const int NO_PROCESSORS);

Eigen::MatrixXd calculate_initial_dissimilarities(
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
void merge_cluster_full(
    std::pair<int, int>& merge,
    std::vector<std::pair<int, int>>& merges,
    std::vector<Cluster>& clusters,
    Eigen::MatrixXd& distance_arr);

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
    Eigen::MatrixXd& distance_arr,
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
    Eigen::MatrixXd& distance_arr,
    double max_merge_distance);

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
    Eigen::MatrixXd& distance_arr);

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
