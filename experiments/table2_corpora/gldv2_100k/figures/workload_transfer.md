# Workload transfer, corpus against corpus -- NOT YET PRODUCED

This comparison needs both corpora and currently has only GLDv2, so no figure
is written. The earlier single-corpus version of these files was removed: it
carried the heading above and the sentence "identical measurement on both
corpora" while listing one, which misreads as a comparison that was never made.

The GLDv2 half is measured and complete, in
`3_method/outputs/workload_transfer/workload_transfer_gldv2_lsq16x4_test_cal5000.{json,md}`.

What is missing is the SOP half. It needs, under `<DATA_ROOT>/sop/`:

    sop_100k_manifest.npz
    sop_100k_dinov2_vits14.npy
    advanced_codec_bundles/lsq16x4_seed{17,42,73,3067,4294,4996,5423,7520,7937,9794}.fcix

and, unless `--allow-recalibration` is passed, the ten matching states under
`nested_additive_states/lsq16x4_seed<N>_ordinal_cal5000.npz`. None of these are
in this repository -- the stored SOP panel reports were produced on another
machine, whose paths they record. With `--allow-recalibration` the states are
rebuilt from the bundles; calibration is deterministic, and the codec hash is
checked against the stored panel report before any work runs, so a mismatched
bundle stops the audit rather than describing a different index.

Then:

    PYTHONPATH=3_method python 4_gldv2/workload_transfer.py measure \
        --data-root <DATA_ROOT> --dataset sop --factory LSQ16x4 --device cuda
    PYTHONPATH=3_method python 4_gldv2/workload_transfer.py compare --factory LSQ16x4

which overwrites this file and writes the three figures.
