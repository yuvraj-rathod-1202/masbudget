"""Export the evaluation split under the difficulty regime as a framework dataset, so the main
comparison can be rerun with the YAML runner (configs/final_mechanism.yaml).

quality = calibrated pass rate of the model on tasks with the same difficulty label (q-hat);
metadata.realized = the task's actual pass/fail per model.

    python experiments/export_datasets.py
"""

from __future__ import annotations

import json

from lib import MEASURED_COSTS, ROOT, PRICINGS, ModelRegistry, calibrate_price, split, view

if __name__ == "__main__":
    calib, evaluation = split()
    tasks = [view(t, "difficulty") for t in evaluation]
    out = {
        "name": "livecodebench_difficulty_eval",
        "description": (
            "Evaluation split (70%, stratified, seed 2026) of data/llmrouterbench_livecodebench.json. quality = "
            "pass rate of the model on calibration-split tasks with the same difficulty label (expected quality "
            "q-hat); metadata.realized = this task's actual pass@1 per model. See experiments/README.md."
        ),
        "tasks": [
            {"task_id": t.task_id, "difficulty": t.difficulty, "category": t.category, "quality": t.quality,
             "metadata": {"realized": t.metadata["realized"], "source_index": t.metadata.get("source_index")}}
            for t in tasks
        ],
    }
    path = ROOT / "data" / "livecodebench_difficulty_eval.json"
    path.write_text(json.dumps(out, indent=1), encoding="utf-8")
    models = ModelRegistry()
    for n, c in PRICINGS["measured"].items():
        models.add(n, c)
    calib_view = [view(t, "difficulty") for t in calib]
    c_min, c_max = min(MEASURED_COSTS.values()), max(MEASURED_COSTS.values())
    for tau in (0.05, 0.2, 0.5):
        per_job = c_min + tau * (c_max - c_min)
        print(f"tau={tau}: budget for 200 jobs = {200 * per_job:.6f}, lambda*_hist = "
              f"{calibrate_price(calib_view, models, per_job * len(calib_view)):.4f}")
    print(f"p_min (floor 0.1) = {0.1 / (c_max - c_min):.4f}; wrote {path} ({len(tasks)} tasks)")
