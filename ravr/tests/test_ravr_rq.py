"""Correctness and budget tests for the RAVR-RQ reference implementation."""
from __future__ import annotations

import os
import json
import struct
import tempfile

import numpy as np

from ravr_rq import (
    HEADER_BYTES,
    LEGACY_OBJECTIVE_PARAMETERS,
    MAGIC,
    OBJECTIVE_LEGACY,
    OBJECTIVE_ORDINAL_KL,
    RAVRRQConfig,
    RAVRRQIndex,
    RAVRRQTrainer,
    evaluate_index,
    evaluate_stage_a_exhaustive,
    solve_exact_small,
    solve_lagrangian,
    _balanced_binary_weights,
    _bernoulli_kl_from_logits,
)


def _toy(seed: int = 4):
    rng = np.random.default_rng(seed)
    d, classes = 24, 8
    centres = rng.normal(size=(classes, d)).astype(np.float32)
    centres /= np.linalg.norm(centres, axis=1, keepdims=True)

    def sample(n):
        labels = rng.integers(classes, size=n)
        scale = np.where(labels[:, None] < classes // 2, 0.12, 0.28)
        x = centres[labels] + scale * rng.normal(size=(n, d))
        x = x.astype(np.float32)
        x /= np.linalg.norm(x, axis=1, keepdims=True)
        return x, labels

    gallery, labels = sample(320)
    calibration, _ = sample(70)
    test, test_labels = sample(40)
    return gallery, labels, calibration, test, test_labels


def _trainer(norm_mode: str = "gram_tables"):
    gallery, labels, calibration, test, test_labels = _toy()
    config = RAVRRQConfig(
        seed=7,
        num_centroids=6,
        nprobe=6,
        stages=4,
        codebook_size=8,
        allowed_lengths=(1, 2, 4),
        base_stages=1,
        rerank_candidates=90,
        retrieval_k=5,
        hard_negative_depth=10,
        kmeans_iterations=5,
        train_samples=300,
        norm_mode=norm_mode,
    )
    trainer = RAVRRQTrainer(config).fit(gallery, calibration)
    return trainer, gallery, labels, test, test_labels


def test_prefix_lookup_and_norm_match_explicit_reconstruction():
    trainer, gallery, _, test, _ = _trainer()
    budget = trainer.minimum_persistent_bytes() + len(gallery) * 2
    index = trainer.build(budget, strategy="boundary")
    explicit = index.dense_reference_scores(test[0])
    ids, lookup = index.search_one(
        test[0], topk=20, nprobe=index.config.num_centroids,
        rerank_candidates=len(gallery)
    )
    np.testing.assert_allclose(lookup, explicit[ids], rtol=2e-5, atol=2e-5)
    # A Gram-table reduction and an explicit vector reduction can round a pair at
    # the cutoff in opposite directions.  Any returned item must nevertheless be
    # within the verified numerical tolerance of the explicit top-20 threshold.
    threshold = np.sort(explicit)[-20]
    assert explicit[ids].min() >= threshold - 2e-5


def test_base_candidate_scan_matches_dense_base_scores():
    trainer, gallery, _, test, _ = _trainer()
    budget = trainer.minimum_persistent_bytes() + len(gallery)
    index = trainer.build(budget, strategy="boundary")
    rows, packed_score, _, _ = index.base_candidates(
        test[1], nprobe=index.config.num_centroids, rerank_candidates=len(gallery)
    )
    ids = np.asarray(index.arrays["item_ids"][rows], dtype=np.int64)
    cfg = trainer.config
    q = test[1]
    coarse = trainer.centroids @ q
    lut = np.einsum("tld,d->tl", trainer.codebooks, q)
    dense = coarse[trainer.cluster_ids].copy()
    for stage in range(cfg.base_stages):
        dense += lut[stage, trainer.full_codes[:, stage].astype(np.int64)]
    dense *= trainer.base_inverse_norm
    np.testing.assert_allclose(packed_score, dense[ids], rtol=8e-4, atol=8e-4)


def test_serialization_round_trip_and_exact_budget():
    trainer, gallery, _, test, _ = _trainer()
    budget = trainer.minimum_persistent_bytes() + len(gallery) * 2
    index = trainer.build(budget, strategy="boundary")
    with tempfile.TemporaryDirectory() as directory:
        path = os.path.join(directory, "toy.rqix")
        index.save(path)
        assert os.path.getsize(path) == index.persistent_bytes
        assert index.persistent_bytes <= budget
        loaded = RAVRRQIndex.load(path)
        before = index.search(test[:8], topk=5)
        after = loaded.search(test[:8], topk=5)
        np.testing.assert_array_equal(before, after)
        assert loaded.resident_bytes == index.resident_bytes
        loaded.close()


def test_index_with_legacy_objective_knobs_still_loads():
    trainer, gallery, _, test, _ = _trainer()
    budget = trainer.minimum_persistent_bytes() + len(gallery) * 2
    index = trainer.build(budget, strategy="boundary")
    with tempfile.TemporaryDirectory() as directory:
        path = os.path.join(directory, "legacy.rqix")
        index.save(path)
        with open(path, "r+b") as handle:
            assert handle.read(8) == MAGIC
            size = struct.unpack("<Q", handle.read(8))[0]
            header = json.loads(handle.read(size))
            header["config"].pop("objective")
            header["config"].update(LEGACY_OBJECTIVE_PARAMETERS)
            header["config"].update({
                "local_density_block_size": 512,
                "local_density_neighbors": 16,
            })
            payload = json.dumps(
                header, sort_keys=True, separators=(",", ":")
            ).encode("utf-8")
            assert len(payload) <= HEADER_BYTES - 16
            handle.seek(8)
            handle.write(struct.pack("<Q", len(payload)))
            handle.write(payload)
            handle.write(b"\0" * (HEADER_BYTES - 16 - len(payload)))
        loaded = RAVRRQIndex.load(path)
        assert loaded.config.objective == OBJECTIVE_LEGACY
        np.testing.assert_array_equal(
            index.search(test[:8], topk=5), loaded.search(test[:8], topk=5)
        )
        loaded.close()


def test_below_minimum_budget_fails_without_output():
    trainer, _, _, _, _ = _trainer()
    try:
        trainer.build(trainer.minimum_persistent_bytes() - 1)
    except ValueError as error:
        assert "minimum legal index size" in str(error)
    else:
        raise AssertionError("an impossible budget unexpectedly succeeded")


def test_allocations_share_codec_and_random_preserves_histogram():
    trainer, gallery, _, _, _ = _trainer()
    budget = trainer.minimum_persistent_bytes() + len(gallery) * 2
    boundary = trainer.build(budget, "boundary")
    random = trainer.build(budget, "random_histogram", random_seed=99)
    fixed = trainer.build(budget, "fixed")
    assert boundary.codec_hash == random.codec_hash == fixed.codec_hash
    assert boundary.allocation.histogram == random.allocation.histogram
    assert boundary.code_only_bytes == random.code_only_bytes
    assert boundary.persistent_bytes <= budget
    assert random.persistent_bytes <= budget
    assert fixed.persistent_bytes <= budget


def test_six_bit_codes_are_actually_packed_and_round_trip():
    gallery, _, calibration, test, _ = _toy()
    config = RAVRRQConfig(
        seed=19,
        num_centroids=6,
        nprobe=6,
        stages=4,
        codebook_size=64,
        allowed_lengths=(2, 3, 4),
        base_stages=2,
        rerank_candidates=90,
        retrieval_k=5,
        hard_negative_depth=10,
        kmeans_iterations=2,
        train_samples=240,
    )
    trainer = RAVRRQTrainer(config).fit(gallery, calibration)
    budget = trainer.fixed_length_budget(3)
    index = trainer.build(budget, "fixed")
    assert np.all(index.assigned_lengths_by_id() == 3)
    assert index.arrays["base_codes"].nbytes == (len(gallery) * 2 * 6 + 7) // 8
    assert index.code_only_bytes_per_item == 18 / 8
    assert index.persistent_bytes == budget
    with tempfile.TemporaryDirectory() as directory:
        path = os.path.join(directory, "six_bit.rqix")
        index.save(path)
        loaded = RAVRRQIndex.load(path)
        np.testing.assert_array_equal(
            index.search(test[:6], topk=5), loaded.search(test[:6], topk=5)
        )
        loaded.close()


def test_lagrangian_monotonicity_and_small_exact_solution():
    # Concave marginal gains make the Lagrangian + repair solution exact here.
    distortion = np.array([
        [9.0, 5.0, 3.0],
        [8.0, 5.5, 4.5],
        [7.0, 4.0, 3.5],
        [6.0, 5.0, 4.2],
    ])
    costs = np.array([0, 1, 2])
    selected = solve_lagrangian(distortion, costs, budget_bytes=4)
    exact = solve_exact_small(distortion, costs, budget_bytes=4)
    got = distortion[np.arange(4), selected].sum()
    optimum = distortion[np.arange(4), exact].sum()
    assert abs(got - optimum) < 1e-12

    usages = []
    for lam in np.linspace(0, 10, 41):
        choices = np.argmin(distortion + lam * costs[None, :], axis=1)
        usages.append(costs[choices].sum())
    assert np.all(np.diff(usages) <= 0)


def test_boundary_kl_is_parameter_free_nonnegative_and_zero_at_teacher():
    teacher = np.array([-1.5, 0.0, 1.25])
    students = np.stack((teacher, teacher + 0.5, teacher - 0.5), axis=1)
    loss = _bernoulli_kl_from_logits(teacher, students)
    assert np.all(loss >= 0)
    np.testing.assert_allclose(loss[:, 0], 0.0, atol=1e-14)
    assert np.all(loss[:, 1:] > 0)

    weights = _balanced_binary_weights(np.array([1, 1, -1, -1, -1]))
    np.testing.assert_allclose(weights[:2].sum(), 0.5)
    np.testing.assert_allclose(weights[2:].sum(), 0.5)
    assert RAVRRQConfig().objective == OBJECTIVE_ORDINAL_KL
    for removed in (
        "lambda_prior", "lambda_score", "lambda_rank",
        "boundary_temperature_ratio", "gamma_min",
    ):
        assert not hasattr(RAVRRQConfig(), removed)


def test_stored_norm_mode_agrees_with_gram_mode():
    gram, gallery, _, test, _ = _trainer("gram_tables")
    stored, _, _, _, _ = _trainer("stored_final_norm")
    gram_budget = gram.fixed_length_budget(2)
    stored_budget = stored.fixed_length_budget(2)
    a = gram.build(gram_budget, "fixed")
    b = stored.build(stored_budget, "fixed")
    ids_a, score_a = a.search(test[:8], topk=8, return_scores=True)
    ids_b, score_b = b.search(test[:8], topk=8, return_scores=True)
    np.testing.assert_array_equal(ids_a, ids_b)
    np.testing.assert_allclose(score_a, score_b, rtol=8e-4, atol=8e-4)


def test_evaluator_reports_required_core_metrics():
    trainer, gallery, labels, test, test_labels = _trainer()
    budget = trainer.minimum_persistent_bytes() + len(gallery) * 2
    index = trainer.build(budget, "boundary")
    report = evaluate_index(
        index, gallery, test, ks=(1, 5),
        gallery_labels=labels, query_labels=test_labels
    )
    for key in (
        "teacher_recall_at_1", "teacher_recall_at_5", "teacher_ndcg_at_5",
        "candidate_recall", "persistent_bytes", "resident_bytes",
        "latency_ms_p95", "qps", "semantic_recall_at_1",
    ):
        assert key in report
    assert 0 <= report["teacher_recall_at_5"] <= 1


def test_validation_role_selects_r_and_self_ids_are_excluded():
    gallery, _, calibration, test, _ = _toy()
    config = RAVRRQConfig(
        seed=13, num_centroids=6, nprobe=6, stages=4, codebook_size=8,
        allowed_lengths=(1, 2, 4), base_stages=1,
        rerank_candidates=160, retrieval_k=5, hard_negative_depth=10,
        kmeans_iterations=4, train_samples=300,
    )
    trainer = RAVRRQTrainer(config).fit(
        gallery, calibration, validation_queries=test
    )
    validation = trainer.validation_report
    assert validation["queries"] == len(test)
    assert validation["selected_rerank_candidates"] <= config.rerank_candidates
    budget = trainer.minimum_persistent_bytes() + len(gallery)
    index = trainer.build(budget)
    ids = index.search(gallery[:12], topk=1, exclude_ids=np.arange(12))
    assert np.all(ids != np.arange(12))


def test_codec_and_allocation_are_deterministic():
    first, gallery, _, test, _ = _trainer()
    second, _, _, _, _ = _trainer()
    assert first.gallery_hash == second.gallery_hash
    assert first.codec_hash == second.codec_hash
    budget = first.minimum_persistent_bytes() + len(gallery) * 2
    a = first.build(budget)
    b = second.build(budget)
    np.testing.assert_array_equal(a.assigned_lengths_by_id(), b.assigned_lengths_by_id())
    np.testing.assert_array_equal(a.search(test, topk=5), b.search(test, topk=5))


def test_cluster_baselines_and_exhaustive_stage_a():
    trainer, gallery, labels, test, test_labels = _trainer()
    budget = trainer.minimum_persistent_bytes() + len(gallery) * 2
    boundary = trainer.build(budget, "boundary")
    shuffled = trainer.build(budget, "within_cluster_random", random_seed=31)
    clustered = trainer.build(budget, "cluster_fixed")
    boundary_length = boundary.assigned_lengths_by_id()
    shuffled_length = shuffled.assigned_lengths_by_id()
    cluster_length = clustered.assigned_lengths_by_id()
    for cluster in range(trainer.config.num_centroids):
        members = trainer.cluster_ids == cluster
        assert sorted(boundary_length[members]) == sorted(shuffled_length[members])
        assert len(np.unique(cluster_length[members])) <= 1
    report = evaluate_stage_a_exhaustive(
        trainer, boundary, test[:10], ks=(1, 5),
        gallery_labels=labels, query_labels=test_labels[:10]
    )
    assert report["evaluation_mode"] == "stage_a_exhaustive"
    assert 0 <= report["teacher_recall_at_5"] <= 1


def test_trainer_state_round_trip():
    trainer, gallery, _, test, _ = _trainer()
    with tempfile.TemporaryDirectory() as directory:
        path = os.path.join(directory, "trainer_state.npz")
        trainer.save_state(path)
        loaded = RAVRRQTrainer.load_state(path, gallery)
        assert loaded.codec_hash == trainer.codec_hash
        assert loaded.config.objective == OBJECTIVE_ORDINAL_KL
        assert loaded.calibration_report["objective"] == OBJECTIVE_ORDINAL_KL
        assert loaded.calibration_report["ordinal_cutoffs"] == [1, 2, 3, 4, 5]
        assert loaded.calibration_report["objective_hyperparameters"] == {}
        np.testing.assert_array_equal(loaded.full_codes, trainer.full_codes)
        budget = trainer.minimum_persistent_bytes() + len(gallery)
        original = trainer.build(budget).search(test[:8], topk=5)
        restored = loaded.build(budget).search(test[:8], topk=5)
        np.testing.assert_array_equal(original, restored)


if __name__ == "__main__":
    tests = [value for name, value in sorted(globals().items()) if name.startswith("test_")]
    for test in tests:
        test()
        print(f"[PASS] {test.__name__}")
