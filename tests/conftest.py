import numpy as np
import pytest
import rasterio
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
                        "data_SAR_HR.mat",
                    ],
                    "labels_file": "TrainImage.mat",
                    "labels_file_test": "TestImage.mat",
                    "na_label": 0,
                    "preprocessing": {"pca": ["data_HS_LR"]},
                }
            ],
            "validation": [],
            "test": [],
        },
        "training": {
            "seed": 42,
            "type": "cv",
            "size": 5,
            "batch_size": 256,
            "epochs": 100,
            "early_stopping": False,
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
                "dropout": 0.4,
            },
        },
        "optimizer": {
            "class_name": "adamw",
            "args": {"lr": 0.001, "weight_decay": 0.0001},
        },
        "metrics": {
            "train": ["accuracy"],
            "test": ["accuracy", "precision", "recall", "f1_score"],
        },
    }


@pytest.fixture
def test_experiment_cfg():
    return {
        "version": "0.0.1",
        "name": "test",
        "run_id": None,
        "description": "REPLICATION OF 'Spatial Gated Multi-Layer Perceptron for Land Use and Land Cover Mapping'",
        "out_dir": None,
        "mlflow_uri": None,
        "datasets": {
            "train": {
                "name": "augsburg",
                "files": {
                    "base_dir": "./data/Datasets/HS-SAR-DSM Augsburg",
                    "feature_files": [
                        "data_DSM.mat",
                        "data_HS_LR.mat",
                        "data_SAR_HR.mat",
                    ],
                    "label_files": ["TrainImage.mat", "TestImage.mat"],
                },
                "preprocessing": {
                    "features_to_process": ["data_HS_LR"],
                    "num_components": 15,
                },
                "na_value": 0,
            },
            "validation": None,
            "test": None,
        },
        "training": {
            "seed": 23253462,
            "type": "cv",
            "n_folds": 3,
            "batch_size": 256,
            "epochs": 1,
            "early_stopping": False,
        },
        "model": {
            "class_name": "litsgumlpmixer",
            "args": {
                "token_features": 16,
                "mixer_features_channel": 16,
                "mixer_features_sequence": 16,
                "dwc_kernels": [1, 3, 5],
                "embedding_kernel_size": 4,
                "num_blocks": 1,
                "activation": "gelu",
                "residual_weight": 2,
                "learnable_residual": False,
                "dropout": 0.4,
            },
        },
        "optimizer": {
            "class_name": "adamw",
            "args": {"lr": 0.001, "weight_decay": 0.0001},
        },
        "metrics": {
            "train": ["accuracy"],
            "test": ["accuracy", "precision", "recall", "f1_score"],
        },
    }


@pytest.fixture
def benchmark_dataset():
    np.random.seed(27182)
    classes = 5
    na_value = 0
    height, width = 224, 224
    channels = 8
    bit_depth = 16

    features = []
    for i in range(3):
        features.append(
            np.random.randint(0, 2**bit_depth, size=(height, width, channels)).astype(
                np.uint16
            )
        )

    targets = np.random.randint(0, classes + 1, (height, width, 1)).astype(np.float32)

    n_pixels = height * width
    idxs = np.array(list(np.ndindex((height, width))))
    np.random.shuffle(idxs)
    idxs_train = tuple(idxs[: n_pixels // 2].T)
    idxs_test = tuple(idxs[n_pixels // 2 :].T)

    targets_train = np.copy(targets)
    targets_train[idxs_test] = 0

    targets_test = np.copy(targets)
    targets_test[idxs_train] = 0

    features = {f"feat{i + 1}": feat for i, feat in enumerate(features)}
    labels = {
        "train": targets_train,
        "test": targets_test,
    }

    return features, labels, na_value


@pytest.fixture
def benchmark_dataset_info(benchmark_dataset, tmp_path):
    features, labels, na_value = benchmark_dataset

    base_dir = tmp_path / "benchmark_dataset"
    base_dir.mkdir(exist_ok=True, parents=True)

    profile = {
        "driver": "GTiff",
        "nodata": na_value,
        "compress": "lzw",
    }

    feature_files = []
    for feat_name, feat in features.items():
        height, width, channels = feat.shape
        profile.update(
            {
                "height": height,
                "width": width,
                "count": channels,
                "dtype": feat.dtype.name,
            }
        )
        filename = f"{feat_name}.tif"
        with rasterio.open(base_dir / filename, "w", **profile) as dst:
            dst.write(np.swapaxes(feat, 0, -1))
        feature_files.append(filename)

    label_files = []
    for split, trgt in labels.items():
        height, width, channels = trgt.shape
        profile.update(
            {
                "height": height,
                "width": width,
                "count": channels,
                "dtype": trgt.dtype.name,
            }
        )
        filename = f"{split}.tif"
        with rasterio.open(base_dir / filename, "w", **profile) as dst:
            dst.write(np.swapaxes(trgt, 0, -1))
        label_files.append(filename)

    return base_dir, feature_files, label_files, na_value
