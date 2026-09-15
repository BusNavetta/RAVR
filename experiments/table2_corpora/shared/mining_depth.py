from __future__ import annotations

import benchmark_nested_additive_ravr as panel

DEFAULT_DEPTH = 50


def tag_for(depth: int) -> str:
    return "" if int(depth) == DEFAULT_DEPTH else f"_hnd{int(depth)}"


def install(depth: int) -> str:
    """Force the mining depth and keep its artifacts separate. Returns the tag."""
    depth = int(depth)
    if depth < 1:
        raise ValueError("hard_negative_depth must be positive")
    tag = tag_for(depth)
    if not tag:
        return tag
    if getattr(panel.calibrate_frozen_additive_codec, "_forced_depth", None) == depth:
        return tag

    original_calibrate = getattr(
        panel.calibrate_frozen_additive_codec, "_original",
        panel.calibrate_frozen_additive_codec,
    )
    original_state = getattr(panel._state_path, "_original", panel._state_path)
    original_dir = getattr(panel.panel_output_dir, "_original",
                           panel.panel_output_dir)

    def calibrate(*args, **kwargs):
        kwargs["hard_negative_depth"] = depth
        return original_calibrate(*args, **kwargs)

    def state_path(data, factory, seed, calibration_queries, precision):
        path = original_state(data, factory, seed, calibration_queries, precision)
        return path.with_name(f"{path.stem}{tag}{path.suffix}")

    def output_dir(dataset):
        return f"{original_dir(dataset)}{tag}"

    for wrapper, original in ((calibrate, original_calibrate),
                              (state_path, original_state),
                              (output_dir, original_dir)):
        wrapper._original = original
    calibrate._forced_depth = depth

    panel.calibrate_frozen_additive_codec = calibrate
    panel._state_path = state_path
    panel.panel_output_dir = output_dir
    # The aggregator imported the resolver into its own namespace.
    import aggregate_nested_additive_ravr as agg
    agg.panel_output_dir = output_dir

    print(f"panel  hard_negative_depth forced to {depth} "
          f"(mined set {panel.ALLOWED_LENGTHS and 10 + 3 * depth} items per "
          f"query, default is {DEFAULT_DEPTH} -> {10 + 3 * DEFAULT_DEPTH}); "
          f"states and outputs carry the suffix '{tag}' so the default-depth "
          "run is untouched", flush=True)
    return tag
