"""Check paired factorial arithmetic and explicit missing C15 cells."""

from importlib import import_module


analysis = import_module("06_experiments.cosyne_2027.analyze_temporal_credit_layers")


def test_paired_factors_and_missing_cells():
    values = {}
    training = {}
    score = {("temporal", "temporal"): 4.0,
             ("temporal", "constant"): 2.0,
             ("constant", "temporal"): 1.0,
             ("constant", "constant"): 0.0}

    def add(family, encoder, worker, seed, value):
        key = family, encoder, worker, seed
        layers = {layer: {metric: value for metric in analysis.METRICS}
                  for layer in analysis.LAYERS}
        values[key] = {"layers": layers, "checkpoint_frames": 100_007_936,
                       "checkpoint_sha256": "checkpoint"}
        training[key] = {"study_id": "study", "study_sha256": "study-hash",
                         "job_id": "42"}

    for seed in analysis.SEEDS:
        add("C01", "temporal", "inapplicable", seed, 3.0)
        add("C01", "constant", "inapplicable", seed, 1.0)
        for (encoder, worker), value in score.items():
            add("C05", encoder, worker, seed, value)
    for seed in (17, 41, 123):
        add("C15", "temporal", "temporal", seed, 2.0)
        add("C15", "constant", "temporal", seed, 1.0)
    for seed in (8, 99):
        add("C15", "temporal", "temporal", seed, 2.0)

    _absolute, _paired, summary, availability = analysis.aggregate(values, training)

    def find(family, factor, fixed):
        return next(row for row in summary if row["family"] == family and
                    row["factor"] == factor and row["fixed_arm"] == fixed and
                    row["layer"] == "ca3" and
                    row["metric"] == "mean_bits_per_activation_eligible")

    assert (find("C01", "encoder", "inapplicable")["n"],
            find("C01", "encoder", "inapplicable")["mean_difference"]) == (5, 2.0)
    assert (find("C05", "interaction", "none")["n"],
            find("C05", "interaction", "none")["mean_difference"]) == (5, 1.0)
    assert (find("C15", "encoder", "temporal")["n"],
            find("C15", "encoder", "temporal")["missing_seeds"]) == (3, "8,99")
    assert find("C15", "worker", "temporal")["n"] == 0
    assert next(row for row in availability if row["family"] == "C15" and
                row["encoder_credit"] == "constant" and
                row["worker_bonus"] == "constant")["missing_panel_seeds"] == "8,17,41,99,123"
