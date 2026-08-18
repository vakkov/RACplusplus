#ifndef RACPP_BUILDING_LIB_ONLY
#define RACPP_BUILDING_LIB_ONLY 0
#endif

#include <array>
#include <tuple>
#include <vector>
#include <set>
#include <limits>
#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <new>
#include "Eigen/Dense"
#include "Eigen/Sparse"

#if defined(__linux__)
#include <sys/mman.h>
#include <unistd.h>
#endif

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
    double nn_distance;
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
using SymDistVector = Eigen::Matrix<SymDistScalar, Eigen::Dynamic, 1>;

// Owning buffer for the O(N^2/2) distance data. Compared to std::vector it:
//  - skips value-initialization: every element is written by dense init
//    before any read, so a fill is pure overhead and concentrates NUMA
//    first-touch on one thread (the parallel tile writers touch first now);
//  - requests transparent hugepages (Linux), cutting TLB misses in the
//    row-hopping NN scans over multi-GB buffers;
//  - returns tail pages to the OS when shrunk (Linux), so RSS drops after
//    each compaction instead of holding the initial peak for the whole run
//    (vector::resize never releases capacity; shrink_to_fit would copy and
//    transiently double RSS).
// Move-only. Non-Linux builds fall back to malloc (no-init still applies;
// shrink keeps memory, matching the old vector behavior).
class SymDistBuffer {
public:
    SymDistBuffer() = default;
    explicit SymDistBuffer(size_t count) { allocate(count); }

    SymDistBuffer(const SymDistBuffer&) = delete;
    SymDistBuffer& operator=(const SymDistBuffer&) = delete;

    SymDistBuffer(SymDistBuffer&& other) noexcept
        : ptr_(other.ptr_), size_(other.size_), mapped_bytes_(other.mapped_bytes_) {
        other.ptr_ = nullptr;
        other.size_ = 0;
        other.mapped_bytes_ = 0;
    }
    SymDistBuffer& operator=(SymDistBuffer&& other) noexcept {
        if (this != &other) {
            release();
            ptr_ = other.ptr_;
            size_ = other.size_;
            mapped_bytes_ = other.mapped_bytes_;
            other.ptr_ = nullptr;
            other.size_ = 0;
            other.mapped_bytes_ = 0;
        }
        return *this;
    }

    ~SymDistBuffer() { release(); }

    SymDistScalar* data() { return ptr_; }
    const SymDistScalar* data() const { return ptr_; }
    size_t size() const { return size_; }

    SymDistScalar& operator[](size_t i) { return ptr_[i]; }
    const SymDistScalar& operator[](size_t i) const { return ptr_[i]; }

    // Shrink preserves the retained prefix and (on Linux) unmaps the tail.
    // Grow preserves existing contents (not used in the current algorithm,
    // kept correct for safety).
    void resize(size_t count) {
        if (count == size_) {
            return;
        }
        if (count < size_) {
#if defined(__linux__)
            const size_t keep_bytes = round_up_page(count * sizeof(SymDistScalar));
            if (ptr_ != nullptr && keep_bytes < mapped_bytes_) {
                if (keep_bytes == 0) {
                    ::munmap(ptr_, mapped_bytes_);
                    ptr_ = nullptr;
                    mapped_bytes_ = 0;
                } else {
                    ::munmap(reinterpret_cast<char*>(ptr_) + keep_bytes,
                             mapped_bytes_ - keep_bytes);
                    mapped_bytes_ = keep_bytes;
                }
            }
#endif
            size_ = count;
            return;
        }
        // Grow: fresh allocation, copy retained prefix.
        SymDistBuffer grown(count);
        if (ptr_ != nullptr && size_ > 0) {
            std::memcpy(grown.ptr_, ptr_, size_ * sizeof(SymDistScalar));
        }
        *this = std::move(grown);
    }

private:
    void allocate(size_t count) {
        size_ = count;
        if (count == 0) {
            return;
        }
        const size_t bytes = count * sizeof(SymDistScalar);
#if defined(__linux__)
        mapped_bytes_ = round_up_page(bytes);
        void* p = ::mmap(nullptr, mapped_bytes_, PROT_READ | PROT_WRITE,
                         MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        if (p == MAP_FAILED) {
            throw std::bad_alloc();
        }
#if defined(MADV_HUGEPAGE)
        ::madvise(p, mapped_bytes_, MADV_HUGEPAGE);  // best-effort
#endif
        ptr_ = static_cast<SymDistScalar*>(p);
#else
        ptr_ = static_cast<SymDistScalar*>(std::malloc(bytes));
        if (ptr_ == nullptr) {
            throw std::bad_alloc();
        }
        mapped_bytes_ = bytes;
#endif
    }

    void release() {
        if (ptr_ != nullptr) {
#if defined(__linux__)
            ::munmap(ptr_, mapped_bytes_);
#else
            std::free(ptr_);
#endif
            ptr_ = nullptr;
        }
        size_ = 0;
        mapped_bytes_ = 0;
    }

#if defined(__linux__)
    static size_t round_up_page(size_t bytes) {
        static const size_t page = static_cast<size_t>(::sysconf(_SC_PAGESIZE));
        return (bytes + page - 1) / page * page;
    }
#endif

    SymDistScalar* ptr_ = nullptr;
    size_t size_ = 0;
    size_t mapped_bytes_ = 0;
};

class SymDistMatrix {
public:
    int N;
    SymDistBuffer data;
    std::vector<size_t> row_start;

    explicit SymDistMatrix(int n)
        : N(n),
          // Uninitialized allocation: dense init writes all N*(N-1)/2 entries
          // before any read (verified by poison-fill equivalence + valgrind).
          data(static_cast<size_t>(n) * (static_cast<size_t>(n) - 1) / 2),
          row_start(static_cast<size_t>(n), 0) {
        for (int i = 0; i < N; i++) {
            row_start[static_cast<size_t>(i)] =
                static_cast<size_t>(i) * static_cast<size_t>(N)
                - static_cast<size_t>(i) * static_cast<size_t>(i + 1) / 2;
        }
    }

    inline size_t tri_idx(int i, int j) const {
        return row_start[static_cast<size_t>(i)]
             + static_cast<size_t>(j - i - 1);
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

    // Fill an existing vector (must already be size N) — no heap allocation.
    template <typename Scalar>
    void get_col_into(int col_id, Eigen::Matrix<Scalar, Eigen::Dynamic, 1>& col) const {
        // k < col_id: scattered access with decreasing stride
        for (int k = 0; k < col_id; ++k) {
            const size_t idx =
                row_start[static_cast<size_t>(k)] +
                static_cast<size_t>(col_id - k - 1);
            col[k] = static_cast<Scalar>(data[idx]);
        }
        col[col_id] = std::numeric_limits<Scalar>::infinity();
        // k > col_id: contiguous access starting at tri_idx(col_id, col_id+1)
        if (col_id + 1 < N) {
            size_t base = row_start[static_cast<size_t>(col_id)];
            for (int k = col_id + 1; k < N; ++k) {
                col[k] = static_cast<Scalar>(data[base + (k - col_id - 1)]);
            }
        }
    }

    template <typename Scalar>
    void set_col(int col_id, const Eigen::Matrix<Scalar, Eigen::Dynamic, 1>& col) {
        // k < col_id: scattered access
        for (int k = 0; k < col_id; ++k) {
            const size_t idx =
                row_start[static_cast<size_t>(k)] +
                static_cast<size_t>(col_id - k - 1);
            data[idx] = static_cast<SymDistScalar>(col[k]);
        }
        // k > col_id: contiguous access
        if (col_id + 1 < N) {
            size_t base = row_start[static_cast<size_t>(col_id)];
            for (int k = col_id + 1; k < N; ++k) {
                data[base + (k - col_id - 1)] = static_cast<SymDistScalar>(col[k]);
            }
        }
    }

    // Write only entries whose counterpart id is in active_ids.
    // This avoids touching dead/secondary rows in hot merge write-back paths.
    template <typename Scalar>
    void set_col_active(
        int col_id,
        const Eigen::Matrix<Scalar, Eigen::Dynamic, 1>& col,
        const std::vector<int>& active_ids) {
        const int* ids = active_ids.data();
        const size_t count = active_ids.size();

        size_t split = 0;
        while (split < count && ids[split] < col_id) {
            ++split;
        }

        // k < col_id: scattered writes.
        for (size_t p = 0; p < split; ++p) {
            const int k = ids[p];
            const size_t idx =
                row_start[static_cast<size_t>(k)] +
                static_cast<size_t>(col_id - k - 1);
            data[idx] = static_cast<SymDistScalar>(col[k]);
        }

        // Skip self entry if present.
        size_t p = split;
        if (p < count && ids[p] == col_id) {
            ++p;
        }

        // k > col_id: write contiguous id runs with contiguous source/destination.
        const size_t col_base = row_start[static_cast<size_t>(col_id)];
        while (p < count) {
            const int k0 = ids[p];
            size_t run_end = p + 1;
            while (run_end < count && ids[run_end] == ids[run_end - 1] + 1) {
                ++run_end;
            }

            const size_t run_len = run_end - p;
            SymDistScalar* dst =
                data.data() + col_base + static_cast<size_t>(k0 - col_id - 1);
            const Scalar* src = col.data() + static_cast<size_t>(k0);

            if constexpr (std::is_same_v<Scalar, SymDistScalar>) {
                std::copy_n(src, run_len, dst);
            } else {
                for (size_t off = 0; off < run_len; ++off) {
                    dst[off] = static_cast<SymDistScalar>(src[off]);
                }
            }
            p = run_end;
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
            const size_t idx =
                row_start[static_cast<size_t>(k)] +
                static_cast<size_t>(col_id - k - 1);
            double v = static_cast<double>(data[idx]);
            if (v < best_val) {
                best_val = v;
                best_idx = k;
            }
        }

        // k > col_id: contiguous access
        if (col_id + 1 < N) {
            size_t base = row_start[static_cast<size_t>(col_id)];
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
            const size_t idx =
                row_start[static_cast<size_t>(k)] +
                static_cast<size_t>(cluster_id - k - 1);
            data[idx] = inf;
        }
        // k > cluster_id: contiguous access
        if (cluster_id + 1 < N) {
            size_t base = row_start[static_cast<size_t>(cluster_id)];
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

void remove_secondary_clusters(
    std::vector<std::pair<int, int> >& merges,
    std::vector<Cluster>& clusters,
    std::vector<int>& active_indices,
    std::vector<int>& active_pos);
//--------------------End Helpers------------------------------------


//-----------------------Distance Calculations-------------------------
//Calculate pairwise cosines between two matrices
Eigen::MatrixXd pairwise_cosine(const Eigen::MatrixXd& A, const Eigen::MatrixXd& B);

//Calculate pairwise euclidean between two matrices
Eigen::MatrixXd pairwise_euclidean(const Eigen::MatrixXd& array_a, const Eigen::MatrixXd& array_b);

// //Averaged dissimilarity across two matrices (wrapper for pairwise distance calc + avging)
// double calculate_weighted_dissimilarity(const Eigen::MatrixXd& points_a, const Eigen::MatrixXd& points_b);

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
    const int NO_PROCESSORS,
    std::vector<int>& dsu_parent,
    std::vector<int>& dsu_size,
    std::vector<SymDistVector>& merged_columns_workspace,
    std::vector<char>& is_iter_secondary_workspace,
    double max_merge_distance,
    const std::vector<char>& is_alive_ws);

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
    const std::vector<std::pair<int, int>>& merges);

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
    const std::vector<std::pair<int, int>>& merges,
    const int NO_PROCESSORS,
    const std::vector<char>& is_alive_ws,
    std::vector<char>& is_dead_ws,
    std::vector<char>& is_changed_ws);

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
    SymDistMatrix& dist,
    std::vector<int>& dsu_parent,
    std::vector<int>& dsu_size);

std::vector<int> RAC(
    const Eigen::MatrixXd& base_arr_in,
    double max_merge_distance,
    Eigen::SparseMatrix<bool>* connectivity,
    int batch_size,
    int no_processors,
    std::string distance_metric);

#if !RACPP_BUILDING_LIB_ONLY
py::array RAC_py(
    py::array base_arr_np,
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
