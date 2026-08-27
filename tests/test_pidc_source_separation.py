from pathlib import Path


def test_pidc_has_no_mimiq_cuda_or_marginal_estimator_dependency():
    cpp = Path(__file__).resolve().parents[1] / "cpp"
    backend = cpp / "pidc_cuda_backend.cu"
    assert backend.is_file()
    assert not (cpp / "cuda_backend.cu").exists()
    assert not (cpp / "histogram_mi.hpp").exists()
    assert not (cpp / "marginals.hpp").exists()

    source = backend.read_text()
    assert "#ifdef" not in source
    assert "#endif" not in source
    assert "mimiq" not in source.lower()
    assert "histogram_mi_kernel" not in source
    assert "run_cuda_pidc" in source
