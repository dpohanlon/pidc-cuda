#pragma once

#include <vector>

template <typename RankT>
struct CompressedPIDCData {
    int n_samples = 0;
    int n_features = 0;
    int rank_stride = 0;
    std::vector<RankT> ranks;  // feature-major: [feature][sample]
    std::vector<int> n_bins;
};
