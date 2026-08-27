#include "pidc_cuda_backend.hpp"

#include <cuda_runtime.h>
#include <cub/block/block_scan.cuh>
#include <cub/device/device_segmented_radix_sort.cuh>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

void check_cuda(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        throw std::runtime_error(
            std::string(operation) + ": " + cudaGetErrorString(status));
    }
}

template <typename T>
class DeviceBuffer {
   public:
    explicit DeviceBuffer(size_t count) : count_(count) {
        if (count_ != 0) {
            check_cuda(
                cudaMalloc(reinterpret_cast<void**>(&ptr_), count_ * sizeof(T)),
                "cudaMalloc");
        }
    }
    ~DeviceBuffer() {
        if (ptr_) cudaFree(ptr_);
    }
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
    T* get() { return ptr_; }
    const T* get() const { return ptr_; }

   private:
    T* ptr_ = nullptr;
    size_t count_ = 0;
};


__device__ __forceinline__ void pair_from_linear(
    unsigned long long pair,
    int features,
    int& i,
    int& j) {
    // Consumer Blackwell has deliberately limited FP64 throughput.  A float
    // square root is sufficient for the initial triangular-row estimate;
    // the integer corrections below make the mapping exact even at a
    // floating-point row boundary.
    const unsigned long long discriminant =
        4ULL * static_cast<unsigned long long>(features) *
            static_cast<unsigned long long>(features - 1) -
        7ULL - 8ULL * pair;
    const float root = sqrtf(static_cast<float>(discriminant));
    i = features - 2 -
        static_cast<int>(floorf(root * 0.5F - 0.5F));
    i = max(0, min(i, features - 2));
    unsigned long long first =
        static_cast<unsigned long long>(i) *
        static_cast<unsigned long long>(2 * features - i - 1) / 2ULL;
    while (i > 0 && pair < first) {
        --i;
        first =
            static_cast<unsigned long long>(i) *
            static_cast<unsigned long long>(2 * features - i - 1) / 2ULL;
    }
    while (i + 1 < features - 1) {
        const unsigned long long next =
            static_cast<unsigned long long>(i + 1) *
            static_cast<unsigned long long>(2 * features - i - 2) / 2ULL;
        if (pair < next) break;
        ++i;
        first = next;
    }
    j = i + 1 + static_cast<int>(pair - first);
}

__host__ __device__ __forceinline__ unsigned long long pair_to_linear(
    int i,
    int j,
    int features) {
    return
        static_cast<unsigned long long>(i) *
            static_cast<unsigned long long>(2 * features - i - 1) / 2ULL +
        static_cast<unsigned long long>(j - i - 1);
}


__global__ void pidc_profile_kernel(
    const uint8_t* __restrict__ ranks,
    const int* __restrict__ n_bins,
    const std::uint32_t* __restrict__ marginal_counts,
    int samples,
    int features,
    int rank_stride,
    int target_begin,
    int targets,
    float* __restrict__ profiles,
    double* __restrict__ mi_out) {

    const int sources = features - 1;
    const unsigned long long ordered_pair =
        static_cast<unsigned long long>(blockIdx.x);
    const int local_target =
        static_cast<int>(ordered_pair / sources);
    if (local_target >= targets) return;
    const int source_index =
        static_cast<int>(ordered_pair % sources);
    const int target = target_begin + local_target;
    const int source =
        source_index >= target ? source_index + 1 : source_index;
    const int target_bins = n_bins[target];
    const int source_bins = n_bins[source];

    extern __shared__ unsigned int joint[];
    const int grid_cells = rank_stride * rank_stride;
    for (int cell = threadIdx.x; cell < grid_cells; cell += blockDim.x) {
        joint[cell] = 0U;
    }
    __syncthreads();

    const uint8_t* target_ranks =
        ranks + static_cast<size_t>(target) * samples;
    const uint8_t* source_ranks =
        ranks + static_cast<size_t>(source) * samples;
    unsigned int low_low = 0U;
    for (int row = threadIdx.x; row < samples; row += blockDim.x) {
        const int target_state = target_ranks[row];
        const int source_state = source_ranks[row];
        if (target_state == 0 && source_state == 0) {
            ++low_low;
        } else {
            atomicAdd(
                joint + target_state * rank_stride + source_state, 1U);
        }
    }
    if (low_low != 0U) atomicAdd(joint, low_low);
    __syncthreads();

    if (threadIdx.x != 0) return;

    const std::uint32_t* target_marginal =
        marginal_counts + static_cast<size_t>(target) * rank_stride;
    const std::uint32_t* source_marginal =
        marginal_counts + static_cast<size_t>(source) * rank_stride;
    double mi = 0.0;
    for (int target_state = 0;
         target_state < target_bins; ++target_state) {
        double specific = 0.0;
        const double target_mass = target_marginal[target_state];
        for (int source_state = 0;
             source_state < source_bins; ++source_state) {
            const unsigned int count =
                joint[target_state * rank_stride + source_state];
            if (count == 0U) continue;
            const double log_ratio = log(
                static_cast<double>(count) * samples /
                (target_mass * source_marginal[source_state]));
            specific += static_cast<double>(count) / target_mass * log_ratio;
        }
        specific = fmax(0.0, specific);
        profiles[
            (static_cast<size_t>(local_target) * rank_stride +
             target_state) *
                sources +
            source_index] = static_cast<float>(specific);
        mi += target_mass / samples * specific;
    }

    if (target < source) {
        mi_out[pair_to_linear(target, source, features)] = mi;
    }
}

__global__ void pidc_all_profiles_kernel(
    const uint8_t* __restrict__ ranks,
    const int* __restrict__ n_bins,
    const std::uint32_t* __restrict__ marginal_counts,
    int samples,
    int features,
    int rank_stride,
    unsigned long long pairs,
    float* __restrict__ profiles,
    double* __restrict__ mi_out) {

    const unsigned long long pair =
        static_cast<unsigned long long>(blockIdx.x);
    if (pair >= pairs) return;
    int feature_x, feature_y;
    pair_from_linear(pair, features, feature_x, feature_y);
    const int sources = features - 1;
    const int nx = n_bins[feature_x];
    const int ny = n_bins[feature_y];

    extern __shared__ unsigned int joint[];
    const int grid_cells = rank_stride * rank_stride;
    for (int cell = threadIdx.x; cell < grid_cells; cell += blockDim.x) {
        joint[cell] = 0U;
    }
    __syncthreads();

    const uint8_t* x =
        ranks + static_cast<size_t>(feature_x) * samples;
    const uint8_t* y =
        ranks + static_cast<size_t>(feature_y) * samples;
    unsigned int low_low = 0U;
    for (int row = threadIdx.x; row < samples; row += blockDim.x) {
        const int rx = x[row];
        const int ry = y[row];
        if (rx == 0 && ry == 0) {
            ++low_low;
        } else {
            atomicAdd(joint + rx * rank_stride + ry, 1U);
        }
    }
    if (low_low != 0U) atomicAdd(joint, low_low);
    __syncthreads();

    if (threadIdx.x != 0) return;
    const std::uint32_t* mx =
        marginal_counts + static_cast<size_t>(feature_x) * rank_stride;
    const std::uint32_t* my =
        marginal_counts + static_cast<size_t>(feature_y) * rank_stride;
    // In the compact source axis, sources after the target shift left by one.
    const int y_as_source_for_x = feature_y - 1;
    const int x_as_source_for_y = feature_x;

    double mi = 0.0;
    for (int ix = 0; ix < nx; ++ix) {
        double specific = 0.0;
        for (int iy = 0; iy < ny; ++iy) {
            const unsigned int count =
                joint[ix * rank_stride + iy];
            if (count == 0U) continue;
            const double log_ratio = log(
                static_cast<double>(count) * samples /
                (static_cast<double>(mx[ix]) * my[iy]));
            specific += static_cast<double>(count) / mx[ix] * log_ratio;
        }
        specific = fmax(0.0, specific);
        profiles[
            (static_cast<size_t>(feature_x) * rank_stride + ix) *
                sources +
            y_as_source_for_x] = static_cast<float>(specific);
        mi += static_cast<double>(mx[ix]) / samples * specific;
    }
    for (int iy = 0; iy < ny; ++iy) {
        double specific = 0.0;
        for (int ix = 0; ix < nx; ++ix) {
            const unsigned int count =
                joint[ix * rank_stride + iy];
            if (count == 0U) continue;
            const double log_ratio = log(
                static_cast<double>(count) * samples /
                (static_cast<double>(mx[ix]) * my[iy]));
            specific += static_cast<double>(count) / my[iy] * log_ratio;
        }
        profiles[
            (static_cast<size_t>(feature_y) * rank_stride + iy) *
                sources +
            x_as_source_for_y] =
            static_cast<float>(fmax(0.0, specific));
    }
    mi_out[pair] = mi;
}

template <int Threads>
__global__ void segmented_prefix_kernel(
    const float* __restrict__ sorted,
    float* __restrict__ prefix,
    int items_per_segment,
    int segments) {

    const int segment = blockIdx.x;
    if (segment >= segments) return;
    using BlockScan = cub::BlockScan<float, Threads>;
    __shared__ typename BlockScan::TempStorage scan_storage;
    __shared__ float carry;
    if (threadIdx.x == 0) carry = 0.0F;
    __syncthreads();

    const size_t segment_start =
        static_cast<size_t>(segment) * items_per_segment;
    for (int begin = 0; begin < items_per_segment; begin += Threads) {
        const int item = begin + threadIdx.x;
        const float value =
            item < items_per_segment
            ? sorted[segment_start + item]
            : 0.0F;
        float inclusive = 0.0F;
        float aggregate = 0.0F;
        BlockScan(scan_storage).InclusiveSum(
            value, inclusive, aggregate);
        const float previous = carry;
        if (item < items_per_segment) {
            prefix[segment_start + item] = previous + inclusive;
        }
        __syncthreads();
        if (threadIdx.x == 0) carry = previous + aggregate;
        __syncthreads();
    }
}

__global__ void pidc_directional_kernel(
    const float* __restrict__ profiles,
    const float* __restrict__ sorted,
    const float* __restrict__ prefix,
    const std::uint32_t* __restrict__ marginal_counts,
    const int* __restrict__ n_bins,
    const double* __restrict__ mi,
    int samples,
    int features,
    int rank_stride,
    int target_begin,
    int targets,
    double* __restrict__ directional) {

    const int sources = features - 1;
    const int ordered_pair =
        blockIdx.x * blockDim.x + threadIdx.x;
    if (ordered_pair >= targets * sources) return;
    const int local_target = ordered_pair / sources;
    const int source_index = ordered_pair % sources;
    const int target = target_begin + local_target;
    const int source =
        source_index >= target ? source_index + 1 : source_index;
    const int left = min(target, source);
    const int right = max(target, source);
    const double pair_mi =
        mi[pair_to_linear(left, right, features)];

    double redundancy_sum = 0.0;
    for (int state = 0; state < n_bins[target]; ++state) {
        const size_t segment =
            static_cast<size_t>(local_target) * rank_stride + state;
        const size_t start = segment * sources;
        const float value = profiles[start + source_index];

        int lower = 0;
        int upper = sources;
        while (lower < upper) {
            const int middle = lower + (upper - lower) / 2;
            if (sorted[start + middle] < value) {
                lower = middle + 1;
            } else {
                upper = middle;
            }
        }
        const int rank = lower;
        const double smaller_sum =
            rank == 0 ? 0.0 : prefix[start + rank - 1];
        const double sum_minimum =
            smaller_sum +
            static_cast<double>(sources - rank - 1) * value;
        const double probability =
            static_cast<double>(
                marginal_counts[
                    static_cast<size_t>(target) * rank_stride + state]) /
            samples;
        redundancy_sum += probability * sum_minimum;
    }

    double score = 0.0;
    if (pair_mi > 0.0 && isfinite(pair_mi)) {
        score =
            static_cast<double>(features - 2) -
            redundancy_sum / pair_mi;
        score = fmin(
            static_cast<double>(features - 2),
            fmax(0.0, score));
    }
    directional[ordered_pair] = score;
}

}  // namespace

bool pidc_cuda_available() {
    int count = 0;
    return cudaGetDeviceCount(&count) == cudaSuccess && count > 0;
}

std::string pidc_cuda_device_name() {
    int device = 0;
    check_cuda(cudaGetDevice(&device), "cudaGetDevice");
    cudaDeviceProp properties{};
    check_cuda(
        cudaGetDeviceProperties(&properties, device),
        "cudaGetDeviceProperties");
    return properties.name;
}


CudaPIDCOutput run_cuda_pidc(
    const CompressedPIDCData<uint8_t>& data,
    const std::vector<std::uint32_t>& marginal_counts,
    int target_batch_size) {

    if (!pidc_cuda_available()) {
        throw std::runtime_error("no CUDA-capable device is available");
    }
    if (data.n_features < 3) {
        throw std::invalid_argument("PIDC requires at least three features");
    }
    if (data.rank_stride <= 0 || data.rank_stride > 64) {
        throw std::invalid_argument(
            "CUDA PIDC requires between 1 and 64 bins");
    }
    if (target_batch_size <= 0) {
        throw std::invalid_argument("target_batch_size must be positive");
    }
    if (marginal_counts.size() !=
        static_cast<size_t>(data.n_features) * data.rank_stride) {
        throw std::invalid_argument("invalid PIDC marginal count array");
    }

    const int features = data.n_features;
    const int sources = features - 1;
    const int max_targets = std::min(target_batch_size, features);
    const int max_segments = max_targets * data.rank_stride;
    const size_t max_items =
        static_cast<size_t>(max_segments) * sources;
    const size_t all_profile_items =
        static_cast<size_t>(features) * data.rank_stride * sources;
    const unsigned long long pair_count =
        static_cast<unsigned long long>(features) *
        static_cast<unsigned long long>(features - 1) / 2ULL;
    if (pair_count >
        static_cast<unsigned long long>(
            std::numeric_limits<unsigned int>::max())) {
        throw std::invalid_argument("too many feature pairs");
    }

    size_t free_bytes = 0;
    size_t total_bytes = 0;
    check_cuda(
        cudaMemGetInfo(&free_bytes, &total_bytes),
        "query CUDA memory for PIDC");
    const size_t estimated_other_bytes =
        data.ranks.size() * sizeof(uint8_t) +
        marginal_counts.size() * sizeof(std::uint32_t) +
        static_cast<size_t>(pair_count) * sizeof(double) +
        4 * max_items * sizeof(float) +
        512ULL * 1024ULL * 1024ULL;
    const bool keep_all_profiles =
        all_profile_items <=
            std::numeric_limits<size_t>::max() / sizeof(float) &&
        all_profile_items * sizeof(float) + estimated_other_bytes <
            free_bytes * 9 / 10;

    DeviceBuffer<uint8_t> d_ranks(data.ranks.size());
    DeviceBuffer<int> d_bins(data.n_bins.size());
    DeviceBuffer<std::uint32_t> d_marginals(marginal_counts.size());
    DeviceBuffer<double> d_mi(static_cast<size_t>(pair_count));
    DeviceBuffer<float> d_all_profiles(
        keep_all_profiles ? all_profile_items : 0);
    DeviceBuffer<float> d_profiles(
        keep_all_profiles ? 0 : max_items);
    DeviceBuffer<float> d_sorted(max_items);
    DeviceBuffer<float> d_prefix(max_items);
    DeviceBuffer<double> d_directional(
        static_cast<size_t>(max_targets) * sources);
    DeviceBuffer<long long> d_offsets(
        static_cast<size_t>(max_segments) + 1);

    check_cuda(
        cudaMemcpy(
            d_ranks.get(), data.ranks.data(),
            data.ranks.size() * sizeof(uint8_t), cudaMemcpyHostToDevice),
        "copy PIDC ranks to CUDA device");
    check_cuda(
        cudaMemcpy(
            d_bins.get(), data.n_bins.data(),
            data.n_bins.size() * sizeof(int), cudaMemcpyHostToDevice),
        "copy PIDC bin counts to CUDA device");
    check_cuda(
        cudaMemcpy(
            d_marginals.get(), marginal_counts.data(),
            marginal_counts.size() * sizeof(std::uint32_t),
            cudaMemcpyHostToDevice),
        "copy PIDC marginals to CUDA device");
    check_cuda(
        cudaMemset(
            d_mi.get(), 0,
            static_cast<size_t>(pair_count) * sizeof(double)),
        "clear PIDC MI output");

    std::vector<long long> offsets(
        static_cast<size_t>(max_segments) + 1);
    for (int segment = 0; segment <= max_segments; ++segment) {
        offsets[segment] =
            static_cast<long long>(segment) * sources;
    }
    check_cuda(
        cudaMemcpy(
            d_offsets.get(), offsets.data(),
            offsets.size() * sizeof(long long), cudaMemcpyHostToDevice),
        "copy PIDC segment offsets");

    size_t sort_temp_bytes = 0;
    const float* sort_query_input =
        keep_all_profiles ? d_all_profiles.get() : d_profiles.get();
    check_cuda(
        cub::DeviceSegmentedRadixSort::SortKeys(
            nullptr, sort_temp_bytes,
            sort_query_input, d_sorted.get(),
            static_cast<long long>(max_items),
            static_cast<long long>(max_segments),
            d_offsets.get(), d_offsets.get() + 1),
        "query PIDC segmented-sort workspace");
    DeviceBuffer<std::uint8_t> d_sort_temp(sort_temp_bytes);

    CudaPIDCOutput output;
    output.mi.resize(static_cast<size_t>(pair_count));
    output.directional.assign(
        static_cast<size_t>(features) * features, 0.0);
    std::vector<double> batch_directional(
        static_cast<size_t>(max_targets) * sources);

    constexpr int threads = 256;
    const size_t shared_bytes =
        static_cast<size_t>(data.rank_stride) * data.rank_stride *
        sizeof(unsigned int);

    if (keep_all_profiles) {
        check_cuda(
            cudaMemset(
                d_all_profiles.get(), 0,
                all_profile_items * sizeof(float)),
            "clear complete PIDC profile array");
        pidc_all_profiles_kernel<<<
            static_cast<unsigned int>(pair_count),
            threads,
            shared_bytes>>>(
            d_ranks.get(), d_bins.get(), d_marginals.get(),
            data.n_samples, features, data.rank_stride,
            pair_count, d_all_profiles.get(), d_mi.get());
        check_cuda(
            cudaGetLastError(),
            "launch complete PIDC profile kernel");
    }

    for (int target_begin = 0;
         target_begin < features;
         target_begin += max_targets) {
        const int targets =
            std::min(max_targets, features - target_begin);
        const int segments = targets * data.rank_stride;
        const size_t items =
            static_cast<size_t>(segments) * sources;
        const unsigned long long ordered_pairs =
            static_cast<unsigned long long>(targets) * sources;
        float* batch_profiles = nullptr;
        if (keep_all_profiles) {
            batch_profiles =
                d_all_profiles.get() +
                static_cast<size_t>(target_begin) *
                    data.rank_stride * sources;
        } else {
            batch_profiles = d_profiles.get();
            check_cuda(
                cudaMemset(
                    batch_profiles, 0, items * sizeof(float)),
                "clear PIDC profile batch");
            pidc_profile_kernel<<<
                static_cast<unsigned int>(ordered_pairs),
                threads,
                shared_bytes>>>(
                d_ranks.get(), d_bins.get(), d_marginals.get(),
                data.n_samples, features, data.rank_stride,
                target_begin, targets, batch_profiles, d_mi.get());
            check_cuda(cudaGetLastError(), "launch PIDC profile kernel");
        }

        size_t current_sort_bytes = sort_temp_bytes;
        check_cuda(
            cub::DeviceSegmentedRadixSort::SortKeys(
                d_sort_temp.get(), current_sort_bytes,
                batch_profiles, d_sorted.get(),
                static_cast<long long>(items),
                static_cast<long long>(segments),
                d_offsets.get(), d_offsets.get() + 1),
            "sort PIDC specific-information segments");

        segmented_prefix_kernel<threads><<<segments, threads>>>(
            d_sorted.get(), d_prefix.get(), sources, segments);
        check_cuda(cudaGetLastError(), "launch PIDC prefix kernel");

        const int aggregate_blocks =
            static_cast<int>((ordered_pairs + threads - 1) / threads);
        pidc_directional_kernel<<<aggregate_blocks, threads>>>(
            batch_profiles, d_sorted.get(), d_prefix.get(),
            d_marginals.get(), d_bins.get(), d_mi.get(),
            data.n_samples, features, data.rank_stride,
            target_begin, targets, d_directional.get());
        check_cuda(cudaGetLastError(), "launch PIDC aggregation kernel");
        check_cuda(cudaDeviceSynchronize(), "execute PIDC CUDA batch");

        check_cuda(
            cudaMemcpy(
                batch_directional.data(), d_directional.get(),
                static_cast<size_t>(targets) * sources * sizeof(double),
                cudaMemcpyDeviceToHost),
            "copy PIDC directional scores");
        for (int local_target = 0;
             local_target < targets; ++local_target) {
            const int target = target_begin + local_target;
            for (int source_index = 0;
                 source_index < sources; ++source_index) {
                const int source =
                    source_index >= target
                    ? source_index + 1
                    : source_index;
                output.directional[
                    static_cast<size_t>(target) * features + source] =
                    batch_directional[
                        static_cast<size_t>(local_target) * sources +
                        source_index];
            }
        }
    }

    check_cuda(
        cudaMemcpy(
            output.mi.data(), d_mi.get(),
            output.mi.size() * sizeof(double), cudaMemcpyDeviceToHost),
        "copy PIDC MI scores");
    return output;
}
