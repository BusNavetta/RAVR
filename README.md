# RAVR — Retrieval-Aware Variable-Rate Allocation

Code and stored results for the paper.

RAVR uses a trained, fixed nested codec and determines how many code stages to keep for each item, while ensuring that the entire serialized index stays within a fixed byte budget. For each item, the cost of keeping a shorter code prefix is estimated using the Ordinal-KL retrieval risk on calibration queries. A dual (Lagrangian) solver then uses these risks to decide how the available budget should be allocated. RAVR-RD uses any remaining budget for items that were not retrieved by any calibration query, based on their reconstruction error.


## Layout

```
/
├── ravr/                  the method and the shared library — no experiments, no results
│   ├── method/
│   │   ├── ravr_rq.py                 Ordinal-KL risk and the dual allocator (solve_lagrangian)
│   │   ├── nested_additive_ravr.py    RAVR on frozen additive codecs (LSQ, RQ) and the packed index
│   │   └── qaq_ravr.py                RAVR on frozen product-quantization codecs (QAQ, QUIP)
│   ├── allocation/
│   │   ├── benchmark_nested_additive_ravr.py   matched-budget panel: uniform, RAVR, RAVR-RD, baselines
│   │   └── aggregate_nested_additive_ravr.py   per-seed reports -> bootstrapped comparison
│   ├── codecs/            Faiss LSQ/RQ training and storage, QAQ and QUIP codecs, search from packed codes
│   ├── common/            dataset loading, exact teacher, metrics, bootstrap, CPU/CUDA backend
│   └── tests/
│
├── tools/                 run once, before the experiments
│   ├── data_prep/         sop/, gldv2_100k/, gldv2_762k/, deep1m/: download and prepare each corpus
│   └── codecs/            benchmark_advanced_codecs.py: trains the frozen LSQ16x4 / RQ16x4 codecs
│
└── experiments/           one folder per result in the paper
    ├── table1_sop/
    ├── fig2_random_allocations/
    ├── table2_corpora/
    └── fig3_native_loss/
```

Each experiment folder holds the code that produced its result, the results
themselves (`outputs/`, `figures/`), and optionally `audits/` — consistency and
robustness checks that are not reported in the paper.


## Where each result comes from

### Table 1 — SOP-100K, LSQ and RQ, 10 seeds
- **Produced by:** `ravr/allocation/benchmark_nested_additive_ravr.py`, then `aggregate_nested_additive_ravr.py`
- **Read from:** `table1_sop/outputs/nested_additive_ravr_sop_100k/evaluation_{lsq16x4,rq16x4}_test_cal5000.md`

### Fig. 2 — random allocations, GLDv2-100K
- **Produced by:** `fig2_random_allocations/run_random_allocations.py`; drawn by `plot_random_allocations.py`
- **Read from:** `random_allocations_gldv2_lsq16x4_seed17.json`, `figures/`

### Table 2 — GLDv2-100K, GLDv2-762K, Deep1M
- **Produced by:** `table2_corpora/<corpus>/run_*.py`, run in stages from codec training to the cross-corpus comparison; `plot_strategies.py`
- **Read from:**
  - `gldv2_100k/figures/sop_vs_gldv2.md`
  - `gldv2_762k/figures/{gain_vs_gallery_size,strategy_frontier}_lsq16x4.md`
  - `deep1m/figures/{gain_across_corpora,strategy_frontier}_lsq16x4.md`

### Fig. 3 — RAVR vs native-loss allocation, QAQ and QUIP
- **Produced by:** in `fig3_native_loss/`, in order:
  1. `prepare_qaq_sop_protocol.py`
  2. `benchmark_{qaq,quip}_official.py`
  3. `benchmark_{qaq,quip}_ravr.py`
  4. `benchmark_native_loss_allocation.py`
  5. `summarize_native_loss_extension.py`

  Drawn by `run_figure3.py`.
- **Read from:** `outputs/native_loss_allocation_10seeds/confirmation.json`, `figures/`



Per-seed reports sit in each experiment's `outputs/`. In `table2_corpora/`,
`shared/` holds helpers used by both large corpora; the other scripts in the
corpus folders (mining width, unseen items, calibration knobs, workload
transfer) are side studies not reported in the paper.

## Running

```bash
pip install -r requirements.txt
python -m pytest -q      
```

Imports are flat (`from ravr_rq import ...`), so put the library on the path.
`pytest.ini` does this for the tests; for scripts:

```bash
export PYTHONPATH="$PWD/ravr/method:$PWD/ravr/allocation:$PWD/ravr/codecs:$PWD/ravr/common"
```

Without any dataset, from the stored reports (run from `Final/`):

```bash
python ravr/allocation/aggregate_nested_additive_ravr.py --dataset sop --factory LSQ16x4   # Table 1 (repeat with RQ16x4)
(cd experiments/table2_corpora/gldv2_762k && python plot_strategies.py)                      # Table 2 frontier
(cd experiments/fig3_native_loss && python run_figure3.py)                                   # Fig. 3
```

The first two rewrite their stored reports in place: the text comes out
identical, while PDF/PNG files may differ with the Matplotlib version.
`run_figure3.py` writes to `generated/` and fails unless every plotted value
matches the stored figure.

Everything else — codec training, calibration, the panels — needs the corpora
prepared by `tools/data_prep/` and a CUDA GPU. Default data locations come from
the original setup (`~/Desktop/AJigma_data`, and absolute `D:\Research\...`
paths in the Fig. 3 drivers), so pass `--data-root` / `--protocol` explicitly.
Results are written into the experiment folder that owns the dataset; set
`RAVR_RESULTS_ROOT` to write elsewhere.

## Reproducibility notes

- Not included: datasets, embeddings, trained codec bundles and calibration
  states. The Fig. 3 audits also need the released QAQ (Julia) and QUIP code.
