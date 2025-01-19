import json

from sgu_mlp.config import ExperimentConfig


def test_experimentconfig(exp_cfg, tmp_path):
    experiment = ExperimentConfig(**exp_cfg)

    assert experiment.to_dict().keys() == exp_cfg.keys()
    assert json.loads(experiment.to_json()).keys() == exp_cfg.keys()

    if exp_cfg["run_id"] == "auto":
        assert experiment.run_id != "auto"

    fp = tmp_path / "experiment.json"
    fp_ = experiment.to_json(fp)
    assert str(fp) == str(fp_)

    experiment_ = ExperimentConfig.from_json(fp_)
    assert experiment == experiment_
