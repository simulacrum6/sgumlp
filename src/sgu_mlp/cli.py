import datetime
import json
import multiprocessing
from pathlib import Path

import gdown
import lightning
import mlflow
import patoolib
import requests
import sklearn.model_selection
import torch
import torchmetrics
from sklearn.model_selection import KFold

from sgu_mlp.config import DatasetConfig
from sgu_mlp.data import Dataset, preprocess
from sgu_mlp.models import LitSGUMLPMixer


def _get_dataloader(dataset, batch_size, idxs=None, shuffle=False, num_workers=1):
    sampler = None
    if idxs is not None:
        sampler = torch.utils.data.SubsetRandomSampler(idxs)
        shuffle = None
    return torch.utils.data.DataLoader(
        dataset,
        num_workers=num_workers,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
    )


def _to_torch_dataset(X, y):
    return torch.utils.data.TensorDataset(torch.from_numpy(X), torch.from_numpy(y))


def _get_logger(experiment_name, run_id, tracking_uri: str | None, save_local: bool):
    if tracking_uri is not None and not save_local:
        try:
            requests.get(tracking_uri + "/health")
        except Exception:
            print("Tracking uri doesn't exist, saving locally to ./ml-runs")
            tracking_uri = None

    return lightning.pytorch.loggers.MLFlowLogger(
        experiment_name=experiment_name, run_name=run_id, tracking_uri=tracking_uri
    )


def _get_metrics(num_labels: int):
    task = "binary" if num_labels == 2 else "multiclass"
    metric_args = dict(task=task, num_classes=num_labels)
    return dict(
        train=dict(
            accuracy=torchmetrics.Accuracy(**metric_args),
        ),
        test=dict(
            accuracy=torchmetrics.Accuracy(**metric_args),
            precision=torchmetrics.Precision(**metric_args),
            recall=torchmetrics.Recall(**metric_args),
            f1=torchmetrics.F1Score(**metric_args),
        ),
    )


def download_datasets():
    data_path = Path("data")
    data_path.mkdir(parents=True, exist_ok=True)
    file_id = "1dLJJrNJpQoQeDHybs37iGxmrSU6aP2xv"  # https://drive.usercontent.google.com/download?id=1dLJJrNJpQoQeDHybs37iGxmrSU6aP2xv&export=download
    file_path = data_path / (file_id + ".rar")
    try:
        gdown.download(id=file_id, output=str(file_path), quiet=False)
        patoolib.extract_archive(str(file_path), outdir=str(data_path))
    finally:
        file_path.unlink(missing_ok=True)


def run_train_test(
    dataloader_train: torch.utils.data.DataLoader,
    model_args: dict,
    trainer_args: dict,
    optimizer_args: dict,
    metrics: dict,
    meta_data: dict,
    dataloader_val: torch.utils.data.DataLoader = None,
    dataloader_test: torch.utils.data.Dataset = None,
    model_class = LitSGUMLPMixer,
):
    model = model_class(model_args, optimizer_args, metrics, meta_data=meta_data)
    trainer = lightning.Trainer(**trainer_args)
    trainer.fit(model, dataloader_train, dataloader_val)
    if dataloader_test:
        trainer.test(model, dataloader_test)


def run_cv(
    dataset_train: torch.utils.data.Dataset,
    model_args: dict,
    trainer_args: dict,
    optimizer_args: dict,
    metrics: dict,
    meta_data: dict,
    dataset_test: torch.utils.data.Dataset = None,
):
    n_folds = meta_data["n_folds"]
    seed = meta_data["seed"]
    batch_size = meta_data["batch_size"]
    cv = KFold(n_splits=n_folds, shuffle=True, random_state=seed)

    test_dataloader = (
        _get_dataloader(dataset_test, batch_size, None, shuffle=False)
        if dataset_test
        else None
    )

    for fold, (train_idx, val_idx) in enumerate(cv.split(dataset_train)):
        train_dataloader = _get_dataloader(
            dataset_train, batch_size, train_idx, shuffle=True
        )
        val_dataloader = _get_dataloader(
            dataset_train, batch_size, val_idx, shuffle=False
        )

        run_train_test(
            train_dataloader,
            model_args,
            trainer_args,
            optimizer_args,
            metrics,
            meta_data,
            val_dataloader,
            test_dataloader,
        )


def setup_experiment(experiment_cfg_path):
    with open(experiment_cfg_path, "r") as f:
        cfg = json.load(f)

    experiment_name = cfg["name"]

    run_id = cfg.get("run_id")
    if run_id is None:
        run_id = f"{experiment_name}__{int(datetime.datetime.now().timestamp())}"

    save_dir = cfg.get("out_dir")
    if save_dir is None:
        save_dir = Path(f"data/runs/{run_id}")
    save_dir.mkdir(parents=True, exist_ok=True)
    with open(save_dir / "config.json", "w") as f:
        json.dump(cfg, f)

    tracking_uri = cfg.get("mlflow_uri")
    save_local = False
    if tracking_uri is None:
        tracking_uri = str(save_dir / "mlflow_logs")
        save_local = True

    logger = _get_logger(
        experiment_name, run_id, tracking_uri=tracking_uri, save_local=save_local
    )

    return (
        cfg,
        run_id,
        save_dir,
        logger,
    )


def cv_experiment(
    experiment_cfg_path: str = "./data/config/replication.experiment.json",
):
    cfg, run_id, save_dir, logger = setup_experiment(experiment_cfg_path)

    experiment_name = cfg["name"]
    training_args = cfg["training"]
    model_args = cfg["model"]["args"]
    optimizer_args = cfg["optimizer"]["args"]
    train_dataset_cfg = cfg["datasets"]["train"]

    lightning.seed_everything(training_args["seed"])

    dataset = Dataset.from_config(DatasetConfig(**train_dataset_cfg))
    X_train, X_test, y_train, y_test = preprocess(dataset)
    dataset_train = torch.utils.data.TensorDataset(
        torch.from_numpy(X_train), torch.from_numpy(y_train)
    )
    dataset_test = torch.utils.data.TensorDataset(
        torch.from_numpy(X_test), torch.from_numpy(y_test)
    )
    model_args["input_dimensions"] = X_train.shape[1:]
    model_args["num_classes"] = dataset.num_labels

    metrics = _get_metrics(dataset.num_labels)

    meta_data = {
        "experiment_name": experiment_name,
        "run_id": run_id,
        "seed": training_args["seed"],
        "batch_size": training_args["batch_size"],
        "n_folds": training_args.get("n_folds", 5),
        "datasets": {
            "train": dataset.name,
            "validation": dataset.name,
        },
    }

    trainer_args = dict(
        default_root_dir=save_dir,
        deterministic=True,
        accelerator="auto",
        max_epochs=training_args["epochs"],
        logger=logger,
    )

    run_cv(
        dataset_train,
        model_args,
        trainer_args,
        optimizer_args,
        metrics,
        meta_data,
        dataset_test,
    )


def ood_experiment():
    # training parameters
    n_epochs = 3
    n_folds = 5
    batch_size = 256
    seed = 271828182
    lightning.seed_everything(seed)
    num_labels = 7

    model_args = dict(
        input_dimensions=(9, 9, 20),
        token_features=256,
        mixer_features_channel=256,
        mixer_features_sequence=256,
        dwc_kernels=(1, 3, 5),
        num_blocks=1,
        activation="gelu",
        residual_weight=2,
        learnable_residual=False,
        embedding_kernel_size=4,
        num_classes=num_labels,
        dropout=0.4,
    )

    optimizer_args = dict(lr=0.001, weight_decay=0.0001)

    metric_args = dict(task="multiclass", num_classes=num_labels)
    metrics = dict(
        train=dict(
            accuracy=torchmetrics.Accuracy(**metric_args),
        ),
        test=dict(
            accuracy=torchmetrics.Accuracy(**metric_args),
            precision=torchmetrics.Precision(**metric_args),
            recall=torchmetrics.Recall(**metric_args),
            f1=torchmetrics.F1Score(**metric_args),
        ),
    )

    # experiment parameters
    run_id = f"cross_location__{int(datetime.datetime.now().timestamp())}"
    save_dir = Path(f"data/runs/{run_id}")
    model_dir = save_dir / "checkpoints"

    mlflow.set_tracking_uri("http://127.0.0.1:8080")
    mlflow.set_experiment(run_id)
    mlflow.autolog()

    # load data
    augsburg = Dataset.from_json(Path("data/config/augsburg.dataset.json"))
    berlin = Dataset.from_json(Path("data/config/augsburg.dataset.json"))

    # todo: change to train only on train portion, add test portion as additional test set
    xy_augsburg = preprocess(augsburg, return_train_test=False)
    xy_berlin = preprocess(berlin, return_train_test=False)

    dataset_pairs = [
        (xy_augsburg, xy_berlin),
        (xy_berlin, xy_augsburg),
    ]

    for i, ((X, y), (X_test, y_test)) in enumerate(dataset_pairs):
        train_percentage = 1 / n_folds
        X_train, X_val, y_train, y_val = sklearn.model_selection.train_test_split(
            X, y, train_size=train_percentage, random_state=seed
        )

        train_dataloader = _get_dataloader(
            _to_torch_dataset(X_train, y_train),
            batch_size=batch_size,
            shuffle=True,
            num_workers=multiprocessing.cpu_count(),
        )
        val_dataloader = _get_dataloader(
            _to_torch_dataset(X_val, y_val),
            batch_size=batch_size,
            shuffle=False,
            num_workers=multiprocessing.cpu_count(),
        )
        test_dataloader = _get_dataloader(
            _to_torch_dataset(X_test, y_test),
            batch_size=batch_size,
            shuffle=False,
            num_workers=multiprocessing.cpu_count(),
        )

        meta_data = {
            "run_id": run_id,
            "seed": seed,
            "datasets": {
                "train": i,
                "validation": i,
                "test": (i + 1) % len(dataset_pairs),
            },
        }
        model = LitSGUMLPMixer(model_args, optimizer_args, metrics, meta_data=meta_data)
        checkpoint_cb = lightning.pytorch.callbacks.ModelCheckpoint(
            filename=f"{i}" + "_{epoch}-{step}",
            save_top_k=1,
            monitor="val_f1",
        )
        trainer = lightning.Trainer(
            default_root_dir=save_dir,
            deterministic=True,
            accelerator="auto",
            max_epochs=n_epochs,
            callbacks=[checkpoint_cb],
        )
        trainer.fit(model, train_dataloader, val_dataloader)
        trainer.test(model, test_dataloader)
