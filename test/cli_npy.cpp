// Standalone verification driver: loads a float32 .npy (N x D, C-order) and
// runs the same float32 no-connectivity path the pybind wrapper uses:
// Map as (D, N) column-major, colwise-normalize, RAC_impl_no_connectivity<float>.
// It #includes the library .cpp directly to reach the static template, so the
// binary always reflects the exact current source (never an installed wheel).
//
// Build (from repo root; point -I at your Eigen checkout):
//   g++ -O3 -std=c++17 -march=native -ffast-math -fopenmp \
//       -I eigen -I src/racplusplus \
//       -DRACPP_BUILDING_LIB_ONLY=1 -DRACPP_SYMDIST_USE_FLOAT=1 \
//       -DRACPP_SIMD_DISSIM_TAIL_UPDATE=1 -DRACPP_SIMD_NN_TAIL_UPDATE=1 \
//       test/cli_npy.cpp -o racpp_npy
//
// Usage: racpp_npy <file.npy> <max_merge_distance> <threads> [row_limit]
// Labels go to stdout (one per line); all library chatter goes to stderr, so
// two builds can be compared with:  diff <(./a ...) <(./b ...)
// Used 2026-08 to verify the SymDistBuffer draft patch is output-identical on
// the 98,496-article embedding set (see test/perf_verification_2026-08.md).
#include "_racplusplus.cpp"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

int main(int argc, char** argv) {
    if (argc < 4) {
        std::fprintf(stderr, "usage: %s file.npy threshold threads [row_limit]\n", argv[0]);
        return 2;
    }
    const char* path = argv[1];
    const double thr = std::atof(argv[2]);
    const int threads = std::atoi(argv[3]);
    const long row_limit = (argc > 4) ? std::atol(argv[4]) : -1;

    std::ifstream f(path, std::ios::binary);
    if (!f) { std::fprintf(stderr, "cannot open %s\n", path); return 2; }
    char magic[8];
    f.read(magic, 8);
    if (std::memcmp(magic, "\x93NUMPY", 6) != 0) { std::fprintf(stderr, "not npy\n"); return 2; }
    uint16_t hlen = 0;
    f.read(reinterpret_cast<char*>(&hlen), 2);
    std::string header(hlen, '\0');
    f.read(header.data(), hlen);
    if (header.find("'<f4'") == std::string::npos ||
        header.find("'fortran_order': False") == std::string::npos) {
        std::fprintf(stderr, "expect little-endian float32 C-order, got: %s\n", header.c_str());
        return 2;
    }
    const size_t sp = header.find("'shape': (");
    long N = 0, D = 0;
    if (sp == std::string::npos ||
        std::sscanf(header.c_str() + sp, "'shape': (%ld, %ld)", &N, &D) != 2) {
        std::fprintf(stderr, "cannot parse shape\n");
        return 2;
    }
    if (row_limit > 0 && row_limit < N) N = row_limit;
    std::fprintf(stderr, "npy: N=%ld D=%ld thr=%g threads=%d\n", N, D, thr, threads);

    std::vector<float> raw(static_cast<size_t>(N) * D);
    f.read(reinterpret_cast<char*>(raw.data()), static_cast<std::streamsize>(raw.size() * 4));
    if (!f) { std::fprintf(stderr, "short read\n"); return 2; }

    // Mirror RAC_py float32 cosine path: (D, N) column-major view of C-order
    // (N, D) data, then colwise normalize into an owned MatrixXf.
    Eigen::Map<const Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic>> base_transposed(
        raw.data(), D, N);
    Eigen::MatrixXf base_arr = base_transposed.colwise().normalized();
    raw.clear();
    raw.shrink_to_fit();

    std::streambuf* cout_buf = std::cout.rdbuf(std::cerr.rdbuf());
    std::vector<int> labels =
        RAC_impl_no_connectivity<float>(base_arr, thr, threads, "cosine");
    std::cout.rdbuf(cout_buf);

    for (int l : labels) std::cout << l << "\n";
    return 0;
}
