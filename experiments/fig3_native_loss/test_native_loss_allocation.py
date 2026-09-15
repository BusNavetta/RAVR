"""Independent direct-sum references for external allocation objectives."""
from types import SimpleNamespace
import numpy as np
import pytest
import torch
from qaq_pq import pack_codes
from benchmark_native_loss_allocation import native_distortions


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA experiment')
def test_native_losses_against_direct_query_sums():
    rng = np.random.default_rng(71)
    x = rng.normal(size=(7, 6)).astype(np.float32)
    q = rng.normal(size=(11, 6)).astype(np.float32)
    x /= np.linalg.norm(x, axis=1, keepdims=True)
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    cb = rng.normal(size=(3, 4, 2)).astype(np.float32)
    codes = rng.integers(0, 4, size=(7, 3), dtype=np.uint8)
    permutation = [2, 4, 1, 5, 3, 0]
    source = SimpleNamespace(metadata={'dimension_permutation': permutation},
                             dimension=6, stages=3, nbits=2,
                             codebooks=cb, packed_codes=pack_codes(codes, 2))
    order = np.array([2, 0, 1])
    allowed = (1, 2, 3)
    got, _ = native_distortions(source, x, q, order, allowed)
    xp, qp = x[:, permutation].astype(float), q[:, permutation].astype(float)
    for i in range(len(x)):
        weight = np.exp(qp @ xp[i] - np.max(qp @ xp[i]))
        weight /= weight.sum()
        reconstruction = np.zeros(6)
        for option, stage in enumerate(order):
            reconstruction[stage*2:stage*2+2] = cb[stage, codes[i, stage]]
            for mode in ('raw', 'cosine'):
                approx = reconstruction if mode == 'raw' else reconstruction / np.linalg.norm(reconstruction)
                error = xp[i] - approx
                squared = np.array([np.dot(query, error)**2 for query in qp])
                block_loss = sum(np.mean([(query[s*2:s*2+2] @ error[s*2:s*2+2])**2
                                          for query in qp]) for s in range(3))
                np.testing.assert_allclose(got['qaq_exact_'+mode][i, option], weight @ squared, rtol=2e-5, atol=2e-7)
                np.testing.assert_allclose(got['quip_block_'+mode][i, option], block_loss, rtol=2e-5, atol=2e-7)
                if mode == 'cosine':
                    np.testing.assert_allclose(got['qip_cosine'][i, option], squared.mean(), rtol=2e-5, atol=2e-7)
