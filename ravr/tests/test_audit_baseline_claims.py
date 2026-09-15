import numpy as np

from audit_baseline_claims import crossed_bootstrap_means, summarize_bootstrap


def test_crossed_bootstrap_is_deterministic_and_keeps_cell_shape():
    values = np.arange(3 * 8 * 4, dtype=np.float64).reshape(3, 8, 4)
    first = crossed_bootstrap_means(values, resamples=50, seed=7, batch_size=11)
    second = crossed_bootstrap_means(values, resamples=50, seed=7, batch_size=11)
    assert first.shape == (50, 4)
    np.testing.assert_array_equal(first, second)


def test_summary_detects_large_positive_and_negative_effects():
    rng = np.random.default_rng(3)
    values = rng.normal(scale=0.01, size=(3, 200, 2))
    values[:, :, 0] += 0.2
    values[:, :, 1] -= 0.2
    boot = crossed_bootstrap_means(values, resamples=2_000, seed=9)
    estimate, se, pointwise, simultaneous, half_width = summarize_bootstrap(values, boot)
    assert estimate[0] > 0 and estimate[1] < 0
    assert np.all(se > 0)
    assert pointwise[0, 0] > 0 and pointwise[1, 1] < 0
    assert simultaneous[0, 0] > 0 and simultaneous[1, 1] < 0
    assert half_width > 0
    np.testing.assert_allclose(simultaneous[:, 1] - estimate, half_width)
