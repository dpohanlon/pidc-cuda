import numpy as np
import pytest

import pidc_cuda


def _correlated_counts(seed=42, cells=4096, genes=14):
    rng = np.random.default_rng(seed)
    factors = rng.normal(size=(cells, 3))
    loadings = rng.normal(size=(3, genes))
    rate = np.exp(0.55 * factors @ loadings + rng.uniform(0.2, 1.2, genes))
    return rng.poisson(rate).astype(np.int32)


def _assert_result(result, genes):
    assert len(result) == 3
    for matrix in result:
        assert matrix.shape == (genes, genes)
        assert np.all(np.isfinite(matrix))
        np.testing.assert_allclose(matrix, matrix.T, atol=0.0, rtol=0.0)
        np.testing.assert_array_equal(np.diag(matrix), 0.0)


def test_cpu_sorted_aggregation_matches_explicit_triplets():
    samples = _correlated_counts(cells=2048, genes=9)
    optimized = pidc_cuda.pidc_cpu(samples, max_bins=8)
    reference = pidc_cuda.pidc_cpu(
        samples, max_bins=8, triplet_reference=True
    )
    _assert_result(optimized, samples.shape[1])
    for actual, expected in zip(optimized, reference):
        np.testing.assert_allclose(actual, expected, rtol=2e-12, atol=2e-12)


@pytest.mark.skipif(
    not pidc_cuda.cuda_available(), reason="CUDA device unavailable"
)
def test_cuda_matches_cpu():
    samples = _correlated_counts(cells=8192, genes=20)
    cpu = pidc_cuda.pidc_cpu(samples, max_bins=8)
    gpu = pidc_cuda.pidc_gpu(
        samples, max_bins=8, target_batch_size=7
    )
    np.testing.assert_allclose(gpu[0], cpu[0], rtol=2e-12, atol=2e-12)
    np.testing.assert_allclose(gpu[1], cpu[1], rtol=2e-6, atol=2e-6)
    np.testing.assert_allclose(gpu[2], cpu[2], rtol=2e-6, atol=2e-6)


def test_rejects_invalid_input():
    with pytest.raises(ValueError, match="three feature"):
        pidc_cuda.pidc_cpu(np.ones((10, 2), dtype=np.int32))
