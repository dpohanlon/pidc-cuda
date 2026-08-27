#include <pybind11/eigen.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <Eigen/Dense>

#include <cstdint>
#include <stdexcept>
#include <string>
#include <utility>

#include "pidc_cuda_backend.hpp"
#include "pidc.hpp"

namespace py = pybind11;

namespace {

using IntMatrixArray =
    py::array_t<int, py::array::c_style | py::array::forcecast>;

Eigen::MatrixXi as_integer_matrix(
    const IntMatrixArray& array, const char* name) {
    if (array.ndim() != 2) {
        throw py::value_error(std::string(name) + " must be two-dimensional");
    }
    const auto input = array.unchecked<2>();
    if (input.shape(0) == 0 || input.shape(1) < 3) {
        throw py::value_error(
            std::string(name) +
            " must contain at least one row and three feature columns");
    }
    Eigen::MatrixXi output(input.shape(0), input.shape(1));
    for (py::ssize_t row = 0; row < input.shape(0); ++row) {
        for (py::ssize_t column = 0; column < input.shape(1); ++column) {
            const int value = input(row, column);
            if (value < 0) {
                throw py::value_error(
                    std::string(name) + " must contain non-negative values");
            }
            output(row, column) = value;
        }
    }
    return output;
}

void check_cpu_bins(int max_bins) {
    if (max_bins < 0 || max_bins > 65535) {
        throw py::value_error(
            "max_bins must be zero or between 1 and 65535");
    }
}

void check_gpu_bins(int max_bins) {
    if (max_bins < 1 || max_bins > 64) {
        throw py::value_error("GPU max_bins must be between 1 and 64");
    }
}

py::tuple result_tuple(PIDCResult result) {
    return py::make_tuple(
        std::move(result.mi),
        std::move(result.puc),
        std::move(result.confidence));
}

PIDCResult unpack_cuda(
    CudaPIDCOutput packed, int features) {
    Eigen::MatrixXd mi = Eigen::MatrixXd::Zero(features, features);
    std::size_t pair = 0;
    for (int left = 0; left < features; ++left) {
        for (int right = left + 1; right < features; ++right, ++pair) {
            mi(left, right) = packed.mi[pair];
            mi(right, left) = packed.mi[pair];
        }
    }
    Eigen::MatrixXd directional(features, features);
    for (int target = 0; target < features; ++target) {
        for (int source = 0; source < features; ++source) {
            directional(target, source) =
                packed.directional[
                    static_cast<std::size_t>(target) * features + source];
        }
    }
    return pidc_detail::finalize(std::move(mi), directional);
}

}  // namespace

PYBIND11_MODULE(_core, module) {
    module.doc() = "Fast C++/CUDA PIDC gene-regulatory-network scoring";

    module.def("cuda_available", &pidc_cuda_available);
    module.def("cuda_device_name", &pidc_cuda_device_name);

    module.def(
        "pidc_cpu",
        [](const IntMatrixArray& samples,
           int max_bins,
           bool triplet_reference) {
            const Eigen::MatrixXi data =
                as_integer_matrix(samples, "samples");
            check_cpu_bins(max_bins);
            PIDCResult result;
            {
                py::gil_scoped_release release;
                result = ::pidc_cpu(data, max_bins, triplet_reference);
            }
            return result_tuple(std::move(result));
        },
        py::arg("samples"),
        py::arg("max_bins") = 16,
        py::arg("triplet_reference") = false,
        "Return (mutual_information, proportional_unique_contribution, "
        "confidence).");

    module.def(
        "pidc_gpu",
        [](const IntMatrixArray& samples,
           int max_bins,
           int target_batch_size) {
            const Eigen::MatrixXi data =
                as_integer_matrix(samples, "samples");
            check_gpu_bins(max_bins);
            if (target_batch_size < 1) {
                throw py::value_error(
                    "target_batch_size must be positive");
            }
            PIDCResult result;
            {
                py::gil_scoped_release release;
                const auto prepared =
                    prepare_pidc_data<std::uint8_t>(data, max_bins);
                result = unpack_cuda(
                    run_cuda_pidc(
                        prepared.data,
                        prepared.marginal_counts,
                        target_batch_size),
                    data.cols());
            }
            return result_tuple(std::move(result));
        },
        py::arg("samples"),
        py::arg("max_bins") = 16,
        py::arg("target_batch_size") = 32,
        "Return (mutual_information, proportional_unique_contribution, "
        "confidence).");
}
