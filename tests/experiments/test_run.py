import json
from pathlib import Path

from experiments.run import cv_experiment


def test_cv_experiment(test_experiment_cfg, tmp_path):
    fp = tmp_path / "cfg.json"
    with open(fp, "w") as f:
        json.dump(test_experiment_cfg, f)

    cv_experiment(str(fp))
