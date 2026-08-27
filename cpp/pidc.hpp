#pragma once

#include <Eigen/Dense>
#include <boost/math/distributions/gamma.hpp>
#include <boost/math/special_functions/digamma.hpp>
#include <boost/math/special_functions/trigamma.hpp>
#include <omp.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

#include "pidc_data.hpp"

// Maximum-likelihood PIDC/PUC for pre-binned non-negative integer data.
//
// The published network score needs pairwise MI and the directional vector
// I_specific(target_state; source).  It does not need a three-dimensional
// histogram.  The optimized aggregation below replaces the explicit context
// gene loop by sorting each (target, target-state) vector and using prefix
// sums to evaluate all sums of pairwise minima.

struct PIDCResult {
    Eigen::MatrixXd mi;
    Eigen::MatrixXd puc;
    Eigen::MatrixXd confidence;
};

template <typename RankT>
struct PIDCPrepared {
    CompressedPIDCData<RankT> data;
    std::vector<std::uint32_t> marginal_counts;
};

template <typename RankT>
PIDCPrepared<RankT> prepare_pidc_data(
    const Eigen::Ref<const Eigen::MatrixXi>& samples,
    int max_bins) {

    PIDCPrepared<RankT> prepared;
    prepared.data = compress_pidc_features<RankT>(samples, max_bins);
    const int features = prepared.data.n_features;
    const int n = prepared.data.n_samples;
    const int stride = prepared.data.rank_stride;
    prepared.marginal_counts.assign(
        static_cast<std::size_t>(features) * stride, 0U);
    for (int feature = 0; feature < features; ++feature) {
        const RankT* ranks =
            prepared.data.ranks.data() +
            static_cast<std::size_t>(feature) * n;
        std::uint32_t* marginal =
            prepared.marginal_counts.data() +
            static_cast<std::size_t>(feature) * stride;
        for (int row = 0; row < n; ++row) {
            ++marginal[static_cast<int>(ranks[row])];
        }
    }
    return prepared;
}

namespace pidc_detail {

struct PairProfiles {
    Eigen::MatrixXd mi;
    // [target][source][target state]
    std::vector<double> specific;
    int features = 0;
    int stride = 0;
};

inline std::size_t specific_offset(
    int target,
    int source,
    int state,
    int features,
    int stride) {
    return (
        (static_cast<std::size_t>(target) * features + source) * stride +
        state);
}

template <typename RankT>
PairProfiles calculate_profiles_cpu(const PIDCPrepared<RankT>& prepared) {
    const auto& data = prepared.data;
    const int n = data.n_samples;
    const int features = data.n_features;
    const int stride = data.rank_stride;

    PairProfiles output;
    output.features = features;
    output.stride = stride;
    output.mi = Eigen::MatrixXd::Zero(features, features);
    output.specific.assign(
        static_cast<std::size_t>(features) * features * stride, 0.0);

    #pragma omp parallel for schedule(dynamic, 1)
    for (int x_feature = 0; x_feature < features; ++x_feature) {
        for (int y_feature = x_feature + 1;
             y_feature < features; ++y_feature) {
            const int nx = data.n_bins[x_feature];
            const int ny = data.n_bins[y_feature];
            const RankT* x =
                data.ranks.data() +
                static_cast<std::size_t>(x_feature) * n;
            const RankT* y =
                data.ranks.data() +
                static_cast<std::size_t>(y_feature) * n;
            const std::uint32_t* mx =
                prepared.marginal_counts.data() +
                static_cast<std::size_t>(x_feature) * stride;
            const std::uint32_t* my =
                prepared.marginal_counts.data() +
                static_cast<std::size_t>(y_feature) * stride;

            std::vector<std::uint32_t> joint(
                static_cast<std::size_t>(nx) * ny, 0U);
            for (int row = 0; row < n; ++row) {
                ++joint[
                    static_cast<std::size_t>(x[row]) * ny +
                    static_cast<int>(y[row])];
            }

            double mi = 0.0;
            for (int ix = 0; ix < nx; ++ix) {
                double specific = 0.0;
                for (int iy = 0; iy < ny; ++iy) {
                    const std::uint32_t count =
                        joint[static_cast<std::size_t>(ix) * ny + iy];
                    if (count == 0U) continue;
                    const double log_ratio = std::log(
                        static_cast<double>(count) * n /
                        (static_cast<double>(mx[ix]) * my[iy]));
                    specific +=
                        static_cast<double>(count) / mx[ix] * log_ratio;
                }
                specific = std::max(0.0, specific);
                output.specific[specific_offset(
                    x_feature, y_feature, ix, features, stride)] = specific;
                mi += static_cast<double>(mx[ix]) / n * specific;
            }

            for (int iy = 0; iy < ny; ++iy) {
                double specific = 0.0;
                for (int ix = 0; ix < nx; ++ix) {
                    const std::uint32_t count =
                        joint[static_cast<std::size_t>(ix) * ny + iy];
                    if (count == 0U) continue;
                    const double log_ratio = std::log(
                        static_cast<double>(count) * n /
                        (static_cast<double>(mx[ix]) * my[iy]));
                    specific +=
                        static_cast<double>(count) / my[iy] * log_ratio;
                }
                output.specific[specific_offset(
                    y_feature, x_feature, iy, features, stride)] =
                    std::max(0.0, specific);
            }

            output.mi(x_feature, y_feature) = mi;
            output.mi(y_feature, x_feature) = mi;
        }
    }
    return output;
}

template <typename RankT>
Eigen::MatrixXd aggregate_sorted(
    const PIDCPrepared<RankT>& prepared,
    const PairProfiles& profiles) {

    const int features = profiles.features;
    const int stride = profiles.stride;
    const int contexts = features - 2;
    const int sources = features - 1;
    Eigen::MatrixXd directional =
        Eigen::MatrixXd::Zero(features, features);

    #pragma omp parallel for schedule(dynamic, 1)
    for (int target = 0; target < features; ++target) {
        std::vector<double> redundancy_sum(features, 0.0);
        std::vector<std::pair<double, int>> ordered;
        ordered.reserve(static_cast<std::size_t>(sources));

        for (int state = 0;
             state < prepared.data.n_bins[target]; ++state) {
            ordered.clear();
            for (int source = 0; source < features; ++source) {
                if (source == target) continue;
                ordered.emplace_back(
                    profiles.specific[specific_offset(
                        target, source, state, features, stride)],
                    source);
            }
            std::sort(
                ordered.begin(), ordered.end(),
                [](const auto& left, const auto& right) {
                    if (left.first != right.first) {
                        return left.first < right.first;
                    }
                    return left.second < right.second;
                });

            const double probability =
                static_cast<double>(
                    prepared.marginal_counts[
                        static_cast<std::size_t>(target) * stride + state]) /
                prepared.data.n_samples;
            double prefix = 0.0;
            for (int rank = 0; rank < sources; ++rank) {
                const double value = ordered[rank].first;
                const double sum_minimum =
                    prefix +
                    static_cast<double>(sources - rank - 1) * value;
                redundancy_sum[ordered[rank].second] +=
                    probability * sum_minimum;
                prefix += value;
            }
        }

        for (int source = 0; source < features; ++source) {
            if (source == target) continue;
            const double mi = profiles.mi(target, source);
            if (mi > 0.0 && std::isfinite(mi)) {
                directional(target, source) = std::clamp(
                    static_cast<double>(contexts) -
                        redundancy_sum[source] / mi,
                    0.0,
                    static_cast<double>(contexts));
            }
        }
    }
    return directional;
}

template <typename RankT>
Eigen::MatrixXd aggregate_triplets(
    const PIDCPrepared<RankT>& prepared,
    const PairProfiles& profiles) {

    const int features = profiles.features;
    const int stride = profiles.stride;
    Eigen::MatrixXd directional =
        Eigen::MatrixXd::Zero(features, features);

    #pragma omp parallel for schedule(dynamic, 1)
    for (int target = 0; target < features; ++target) {
        for (int source = 0; source < features; ++source) {
            if (source == target) continue;
            const double mi = profiles.mi(target, source);
            if (!(mi > 0.0) || !std::isfinite(mi)) continue;
            double score = 0.0;
            for (int context = 0; context < features; ++context) {
                if (context == target || context == source) continue;
                double redundancy = 0.0;
                for (int state = 0;
                     state < prepared.data.n_bins[target]; ++state) {
                    const double probability =
                        static_cast<double>(
                            prepared.marginal_counts[
                                static_cast<std::size_t>(target) * stride +
                                state]) /
                        prepared.data.n_samples;
                    const double source_specific =
                        profiles.specific[specific_offset(
                            target, source, state, features, stride)];
                    const double context_specific =
                        profiles.specific[specific_offset(
                            target, context, state, features, stride)];
                    redundancy += probability *
                        std::min(source_specific, context_specific);
                }
                const double ratio = (mi - redundancy) / mi;
                if (std::isfinite(ratio) && ratio >= 0.0) {
                    score += ratio;
                }
            }
            directional(target, source) = score;
        }
    }
    return directional;
}

inline Eigen::MatrixXd symmetric_puc(
    const Eigen::Ref<const Eigen::MatrixXd>& directional) {
    const int features = directional.rows();
    Eigen::MatrixXd puc = Eigen::MatrixXd::Zero(features, features);
    for (int i = 0; i < features; ++i) {
        for (int j = i + 1; j < features; ++j) {
            const double value = directional(i, j) + directional(j, i);
            puc(i, j) = value;
            puc(j, i) = value;
        }
    }
    return puc;
}

struct ContextFit {
    bool gamma_valid = false;
    double shape = 0.0;
    double scale = 0.0;
    double mean = 0.0;
    double variance = 0.0;
};

inline ContextFit fit_context_distribution(
    const Eigen::Ref<const Eigen::MatrixXd>& puc,
    int feature) {
    const int features = puc.rows();
    ContextFit result;
    std::vector<double> values;
    values.reserve(static_cast<std::size_t>(features - 1));
    double sum = 0.0;
    double sum_log = 0.0;
    bool all_positive = true;
    for (int other = 0; other < features; ++other) {
        if (other == feature) continue;
        const double value = puc(feature, other);
        values.push_back(value);
        sum += value;
        if (value > 0.0 && std::isfinite(value)) {
            sum_log += std::log(value);
        } else {
            all_positive = false;
        }
    }
    const int count = static_cast<int>(values.size());
    result.mean = sum / count;
    double squared = 0.0;
    for (double value : values) {
        const double difference = value - result.mean;
        squared += difference * difference;
    }
    result.variance = count > 1 ? squared / (count - 1) : 0.0;

    if (!all_positive || !(result.mean > 0.0)) return result;
    const double s = std::log(result.mean) - sum_log / count;
    if (!(s > 1e-14) || !std::isfinite(s)) return result;

    double shape =
        (3.0 - s + std::sqrt((s - 3.0) * (s - 3.0) + 24.0 * s)) /
        (12.0 * s);
    for (int iteration = 0; iteration < 30; ++iteration) {
        const double value =
            std::log(shape) - boost::math::digamma(shape) - s;
        const double derivative =
            1.0 / shape - boost::math::trigamma(shape);
        const double update = value / derivative;
        const double candidate = shape - update;
        if (!(candidate > 0.0) || !std::isfinite(candidate)) break;
        shape = candidate;
        if (std::abs(update) <= 1e-12 * shape) break;
    }
    const double scale = result.mean / shape;
    if (shape > 0.0 && scale > 0.0 &&
        std::isfinite(shape) && std::isfinite(scale)) {
        result.gamma_valid = true;
        result.shape = shape;
        result.scale = scale;
    }
    return result;
}

inline Eigen::MatrixXd apply_pidc_context(
    const Eigen::Ref<const Eigen::MatrixXd>& puc) {
    const int features = puc.rows();
    std::vector<ContextFit> fits(features);
    #pragma omp parallel for schedule(static)
    for (int feature = 0; feature < features; ++feature) {
        fits[feature] = fit_context_distribution(puc, feature);
    }

    Eigen::MatrixXd confidence =
        Eigen::MatrixXd::Zero(features, features);
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < features; ++i) {
        for (int j = i + 1; j < features; ++j) {
            const double score = puc(i, j);
            double value = 0.0;
            if (fits[i].gamma_valid && fits[j].gamma_valid) {
                try {
                    const boost::math::gamma_distribution<double> dist_i(
                        fits[i].shape, fits[i].scale);
                    const boost::math::gamma_distribution<double> dist_j(
                        fits[j].shape, fits[j].scale);
                    value =
                        boost::math::cdf(dist_i, score) +
                        boost::math::cdf(dist_j, score);
                } catch (...) {
                    value = 0.0;
                }
            }
            if (!(fits[i].gamma_valid && fits[j].gamma_valid) ||
                !std::isfinite(value)) {
                const double di = score - fits[i].mean;
                const double dj = score - fits[j].mean;
                const double zi =
                    fits[i].variance > 0.0 && di > 0.0
                    ? di * di / fits[i].variance
                    : 0.0;
                const double zj =
                    fits[j].variance > 0.0 && dj > 0.0
                    ? dj * dj / fits[j].variance
                    : 0.0;
                value = std::sqrt(zi + zj);
            }
            confidence(i, j) = value;
            confidence(j, i) = value;
        }
    }
    return confidence;
}

inline PIDCResult finalize(
    Eigen::MatrixXd mi,
    const Eigen::Ref<const Eigen::MatrixXd>& directional) {
    PIDCResult result;
    result.mi = std::move(mi);
    result.puc = symmetric_puc(directional);
    result.confidence = apply_pidc_context(result.puc);
    return result;
}

}  // namespace pidc_detail

inline PIDCResult pidc_cpu(
    const Eigen::Ref<const Eigen::MatrixXi>& samples,
    int max_bins = 16,
    bool triplet_reference = false) {
    if (samples.cols() < 3) {
        throw std::invalid_argument("PIDC requires at least three features");
    }
    if (max_bins < 0 || max_bins > 65535) {
        throw std::invalid_argument(
            "max_bins must be zero or between 1 and 65535");
    }
    const auto prepared =
        prepare_pidc_data<std::uint16_t>(samples, max_bins);
    const auto profiles = pidc_detail::calculate_profiles_cpu(prepared);
    const Eigen::MatrixXd directional = triplet_reference
        ? pidc_detail::aggregate_triplets(prepared, profiles)
        : pidc_detail::aggregate_sorted(prepared, profiles);
    return pidc_detail::finalize(profiles.mi, directional);
}
