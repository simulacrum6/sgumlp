from sgu_mlp.config import ExperimentConfig


def test_experimentconfig(exp_cfg):
    experiment = ExperimentConfig(**exp_cfg)
    print(experiment)