import pytest
import torch


@pytest.fixture
def hs_image():
    return torch.randn(2, 16, 128, 128)


@pytest.fixture
def tokens():
    return torch.randn(2, 81, 768)


@pytest.fixture
def image_and_labels():
    image = torch.rand((128, 128, 16)).numpy()
    labels = torch.randint(0, 10, (128, 128)).long().numpy()
    return image, labels


@pytest.fixture
def patches():
    return torch.randn(768, 9, 9, 32)


@pytest.fixture
def dataset():
    n = 64
    p = 17
    c = 32
    k = 10
    X = torch.randn(n, p, p, c)
    y = torch.randint(0, k, (n,)).long()
    return X, y, k


@pytest.fixture
def sgumlpmixer_args():
    return dict(
        token_features=256,
        mixer_features_channel=768,
        mixer_features_sequence=768,
        num_blocks=2,
        activation="relu",
    )


@pytest.fixture
def adamw_args():
    return dict(
        lr=1e-3,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.01,
    )

@pytest.fixture
def exp_cfg():
    return {
        "version": "0.0.1",
        "run_id": "auto",
        "name": "REPLICATION OF 'Spatial Gated Multi-Layer Perceptron for Land Use and Land Cover Mapping'",
        "datasets": {
            "train": [
                {
                    "name": "augsburg",
                    "base_dir": "./data/datasets/HS-SAR-DSM Augsburg",
                    "feature_files": [
                        "data_DSM.mat",
                        "data_HS_LR.mat",
                        "data_SAR_HR.mat"
                    ],
                    "labels_file": "TrainImage.mat",
                    "labels_file_test": "TestImage.mat",
                    "na_label": 0,
                    "preprocessing": {
                        "pca": ["data_HS_LR"]
                    }
                }
            ],
            "validation": [],
            "test": []
        },
        "training": {
            "seed": 42,
            "type": "cv",
            "size": 5,
            "batch_size": 256,
            "epochs": 100,
            "early_stopping": False
        },
        "model": {
            "class_name": "litsgumlpmixer",
            "args": {
                "input_dimensions": -1,
                "num_classes": -1,
                "token_features": 256,
                "mixer_features_channel": 256,
                "mixer_features_sequence": 256,
                "dwc_kernels": [1, 3, 5],
                "embedding_kernel_size": 4,
                "num_blocks": 1,
                "activation": "gelu",
                "residual_weight": 2,
                "learnable_residual": False,
                "dropout": 0.4
            }
        },
        "optimizer": {
            "class_name": "adamw",
            "args": {
                "lr": 0.001,
                "weight_decay": 0.0001
            }
        },
        "metrics": {
            "train": [
                "accuracy"
            ],
            "test": [
                "accuracy",
                "precision",
                "recall",
                "f1_score"
            ]
        }
    }