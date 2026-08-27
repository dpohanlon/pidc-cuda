#!/usr/bin/env python3
"""Reproducible CPU/GPU PIDC timing and equivalence benchmark."""

import argparse
import statistics
import time

import numpy as np

import pidc_cuda


def timed(function, repeats):
    values = []
    result = None
    for _ in range(repeats):
        start = time.perf_counter()
        result = function()
        values.append(time.perf_counter() - start)
    return result, values


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cells", type=int, default=20_000)
    parser.add_argument("--genes", type=int, default=1_000)
    parser.add_argument("--bins", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20260725)
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    factors = rng.normal(size=(args.cells, 8))
    loadings = rng.normal(scale=0.3, size=(8, args.genes))
    baseline = rng.uniform(-0.5, 1.5, args.genes)
    rate = np.exp(factors @ loadings + baseline)
    samples = rng.poisson(rate).astype(np.int32)

    gpu_fn = lambda: pidc_cuda.pidc_gpu(
        samples,
        max_bins=args.bins,
        target_batch_size=args.batch_size,
    )
    pidc_cuda.pidc_gpu(
        samples[: min(args.cells, 2048), : min(args.genes, 64)],
        max_bins=args.bins,
        target_batch_size=min(args.batch_size, 32),
    )
    gpu, gpu_times = timed(gpu_fn, args.repeats)
    print(
        f"GPU median={statistics.median(gpu_times):.6f}s "
        f"runs={gpu_times}"
    )

    if args.cpu:
        cpu, cpu_times = timed(
            lambda: pidc_cuda.pidc_cpu(samples, max_bins=args.bins),
            args.repeats,
        )
        errors = [
            float(np.max(np.abs(actual - expected)))
            for actual, expected in zip(gpu, cpu)
        ]
        print(
            f"CPU median={statistics.median(cpu_times):.6f}s "
            f"runs={cpu_times}"
        )
        print(f"max_abs_errors={errors}")


if __name__ == "__main__":
    main()
