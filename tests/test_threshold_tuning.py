from __future__ import annotations

import json


def test_threshold_bundle_has_sweep_and_selection(trained_bundle_dir):
    payload = json.loads((trained_bundle_dir / "threshold.json").read_text())

    assert payload["selection_metric"] == "f1"
    assert payload["tuned_on"] == "validation"

    sweep = payload["sweep"]
    assert len(sweep) >= 2
    for row in sweep:
        for key in ("threshold", "precision", "recall", "f1"):
            assert key in row
            assert 0.0 <= row[key] <= 1.0

    selected = payload["decision_threshold"]
    best = max(sweep, key=lambda r: r["f1"])
    assert selected == best["threshold"]
