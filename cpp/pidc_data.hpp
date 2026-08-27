#pragma once

#include <Eigen/Dense>
#include <omp.h>

#include <algorithm>
#include <cstdint>
#include <functional>
#include <limits>
#include <queue>
#include <stdexcept>
#include <unordered_map>
#include <vector>

#include "pidc_compressed_data.hpp"

// PIDC-specific order compression. PIDC only needs empirical state ranks and
// counts; it has no fitted-marginal CDF boundaries and no dependency on an
// adaptive pairwise-MI estimator.
template <typename RankT>
CompressedPIDCData<RankT> compress_pidc_features(
    const Eigen::Ref<const Eigen::MatrixXi>& samples,
    int max_bins) {

    const int n_samples = samples.rows();
    const int n_features = samples.cols();
    const int rank_limit =
        static_cast<int>(std::numeric_limits<RankT>::max()) + 1;
    if (max_bins < 0 || max_bins > rank_limit) {
        throw std::invalid_argument("max_bins exceeds the rank representation");
    }

    struct FeatureEncoding {
        std::vector<RankT> ranks;
        int bins = 0;
    };
    std::vector<FeatureEncoding> encoded(n_features);
    int observed_max_bins = 0;

    #pragma omp parallel for schedule(static) reduction(max:observed_max_bins)
    for (int feature = 0; feature < n_features; ++feature) {
        int max_value = 0;
        for (int row = 0; row < n_samples; ++row) {
            max_value = std::max(max_value, samples(row, feature));
        }

        const bool use_direct =
            max_value <= 1'000'000 &&
            static_cast<long long>(max_value) <=
                8LL * static_cast<long long>(n_samples) + 4096LL;

        std::vector<int> values;
        std::vector<int> frequencies;
        if (use_direct) {
            std::vector<int> frequency(
                static_cast<std::size_t>(max_value) + 1, 0);
            for (int row = 0; row < n_samples; ++row) {
                ++frequency[static_cast<std::size_t>(samples(row, feature))];
            }
            for (int value = 0; value <= max_value; ++value) {
                const int count = frequency[static_cast<std::size_t>(value)];
                if (count != 0) {
                    values.push_back(value);
                    frequencies.push_back(count);
                }
            }
        } else {
            std::unordered_map<int, int> frequency;
            frequency.reserve(static_cast<std::size_t>(n_samples / 2 + 1));
            for (int row = 0; row < n_samples; ++row) {
                ++frequency[samples(row, feature)];
            }
            values.reserve(frequency.size());
            for (const auto& entry : frequency) values.push_back(entry.first);
            std::sort(values.begin(), values.end());
            frequencies.reserve(values.size());
            for (int value : values) frequencies.push_back(frequency[value]);
        }

        const int unique_count = static_cast<int>(values.size());
        const bool bounded = max_bins != 0 && unique_count > max_bins;
        int bins = bounded ? max_bins : unique_count;
        if (bins > rank_limit) {
            encoded[feature].bins = -1;
            continue;
        }

        std::vector<int> unique_to_bin(unique_count);
        if (!bounded) {
            for (int state = 0; state < unique_count; ++state) {
                unique_to_bin[state] = state;
            }
        } else {
            struct Group {
                long long mass;
                int first;
                int last;
                int previous;
                int next;
                unsigned int version;
                bool active;
            };
            struct MergeCandidate {
                long long mass;
                int left;
                int right;
                unsigned int left_version;
                unsigned int right_version;
                bool operator>(const MergeCandidate& other) const {
                    return mass > other.mass;
                }
            };

            std::vector<Group> groups(unique_count);
            for (int state = 0; state < unique_count; ++state) {
                groups[state] = Group{
                    frequencies[state],
                    state,
                    state,
                    state - 1,
                    state + 1 < unique_count ? state + 1 : -1,
                    0U,
                    true};
            }

            std::priority_queue<
                MergeCandidate,
                std::vector<MergeCandidate>,
                std::greater<MergeCandidate>>
                candidates;
            const auto enqueue = [&](int left) {
                if (left < 0 || !groups[left].active) return;
                const int right = groups[left].next;
                if (right < 0 || !groups[right].active) return;
                candidates.push(MergeCandidate{
                    groups[left].mass + groups[right].mass,
                    left,
                    right,
                    groups[left].version,
                    groups[right].version});
            };
            for (int state = 0; state + 1 < unique_count; ++state) {
                enqueue(state);
            }

            int active_groups = unique_count;
            while (active_groups > max_bins) {
                MergeCandidate candidate;
                do {
                    if (candidates.empty()) {
                        throw std::runtime_error(
                            "failed to construct bounded PIDC states");
                    }
                    candidate = candidates.top();
                    candidates.pop();
                } while (
                    !groups[candidate.left].active ||
                    !groups[candidate.right].active ||
                    groups[candidate.left].next != candidate.right ||
                    groups[candidate.left].version !=
                        candidate.left_version ||
                    groups[candidate.right].version !=
                        candidate.right_version);

                Group& left = groups[candidate.left];
                Group& right = groups[candidate.right];
                const int previous = left.previous;
                const int next = right.next;
                left.mass += right.mass;
                left.last = right.last;
                left.next = next;
                ++left.version;
                right.active = false;
                ++right.version;
                if (next >= 0) {
                    groups[next].previous = candidate.left;
                    ++groups[next].version;
                }
                --active_groups;
                enqueue(previous);
                enqueue(candidate.left);
                enqueue(next);
            }

            int group = 0;
            int current = 0;
            while (groups[current].previous >= 0) {
                current = groups[current].previous;
            }
            while (current >= 0) {
                for (int state = groups[current].first;
                     state <= groups[current].last;
                     ++state) {
                    unique_to_bin[state] = group;
                }
                ++group;
                current = groups[current].next;
            }
            bins = group;
        }

        auto& output = encoded[feature];
        output.bins = bins;
        output.ranks.resize(n_samples);
        if (use_direct) {
            std::vector<RankT> value_to_bin(
                static_cast<std::size_t>(max_value) + 1, RankT{0});
            for (int state = 0; state < unique_count; ++state) {
                value_to_bin[static_cast<std::size_t>(values[state])] =
                    static_cast<RankT>(unique_to_bin[state]);
            }
            for (int row = 0; row < n_samples; ++row) {
                output.ranks[row] =
                    value_to_bin[
                        static_cast<std::size_t>(samples(row, feature))];
            }
        } else {
            std::unordered_map<int, RankT> value_to_bin;
            value_to_bin.reserve(values.size());
            for (int state = 0; state < unique_count; ++state) {
                value_to_bin.emplace(
                    values[state],
                    static_cast<RankT>(unique_to_bin[state]));
            }
            for (int row = 0; row < n_samples; ++row) {
                output.ranks[row] = value_to_bin.at(samples(row, feature));
            }
        }
        observed_max_bins = std::max(observed_max_bins, bins);
    }

    CompressedPIDCData<RankT> result;
    result.n_samples = n_samples;
    result.n_features = n_features;
    result.rank_stride = observed_max_bins;
    result.ranks.resize(
        static_cast<std::size_t>(n_features) * n_samples);
    result.n_bins.resize(n_features);
    for (int feature = 0; feature < n_features; ++feature) {
        const auto& input = encoded[feature];
        if (input.bins < 0) {
            throw std::runtime_error(
                "feature has too many distinct values for rank type");
        }
        std::copy(
            input.ranks.begin(),
            input.ranks.end(),
            result.ranks.begin() +
                static_cast<std::size_t>(feature) * n_samples);
        result.n_bins[feature] = input.bins;
    }
    return result;
}
