#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "pidc_compressed_data.hpp"

struct CudaPIDCOutput {
    // Pairwise MI in strict-upper-triangle linear order.
    std::vector<double> mi;
    // Dense directional PUC sums: [target][source].
    std::vector<double> directional;
};

bool pidc_cuda_available();
std::string pidc_cuda_device_name();

CudaPIDCOutput run_cuda_pidc(
    const CompressedPIDCData<uint8_t>& data,
    const std::vector<std::uint32_t>& marginal_counts,
    int target_batch_size);
