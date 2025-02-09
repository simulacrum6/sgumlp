import datetime
import json
import multiprocessing
from pathlib import Path

import gdown
import lightning
import numpy as np
import pandas as pd
import patoolib
import requests
import torch
import torchmetrics
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader

from sgu_mlp.data import load_benchmark_dataset, preprocess, PatchDataset
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
    dataloader_test: torch.utils.data.Dataset | list = None,
    model_class=LitSGUMLPMixer,
    criterion=None,
):
    model = model_class(
        model_args, optimizer_args, metrics, criterion=criterion, meta_data=meta_data
    )
    trainer = lightning.Trainer(**trainer_args)
    trainer.fit(model, dataloader_train, dataloader_val)
    if dataloader_test is None:
        return
    if type(dataloader_test) is list:
        for dl in dataloader_test:
            trainer.test(model, dl)
    else:
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

    print(f"Running experiment '{experiment_name}' ({run_id})")
    print(f"logging results to {tracking_uri}")
    print("config:")
    print(cfg)

    return (
        cfg,
        run_id,
        save_dir,
        logger,
    )


def _load_and_preprocess_dataset(dataset_cfg: dict, pcas=None):
    dataset = load_benchmark_dataset(
        **dataset_cfg["files"], na_value=dataset_cfg.get("na_value", 0)
    )
    X, y, (train_idx, test_idx), label_encoder, pcas = preprocess(
        *dataset, **dataset_cfg["preprocessing"], pcas=pcas
    )
    n, p, _, c = X.shape
    X_train = X[train_idx]
    y_train = y[train_idx]
    X_test = X[test_idx]
    y_test = y[test_idx]
    dataset_train = torch.utils.data.TensorDataset(
        torch.from_numpy(X_train), torch.from_numpy(y_train)
    )
    dataset_test = torch.utils.data.TensorDataset(
        torch.from_numpy(X_test), torch.from_numpy(y_test)
    )

    return {
        "name": dataset_cfg["name"],
        "train": dataset_train,
        "test": dataset_test,
        "num_labels": len(label_encoder.classes_),
        "input_dimensions": X_train.shape[1:],
        "pcas": pcas,
        "label_encoder": label_encoder,
    }


class CustomCosineSimilarity(torchmetrics.CosineSimilarity):
    def update(self, preds, target):
        target = torch.nan_to_num(target, 0.0)

        eps = 1e-8
        target = target + eps

        target = target / target.sum(1, keepdim=True)

        preds_proba = torch.softmax(preds, 1) + eps
        preds_proba = preds_proba / preds_proba.sum(1, keepdim=True)
        super().update(preds_proba, target)


class CustomKLDivergence(torchmetrics.KLDivergence):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.log_prob = False

    def update(self, preds, target):
        target = torch.nan_to_num(target, 0.0)
        eps = 1e-8
        target = target + eps
        target = target / target.sum(1, keepdim=True)
        preds_proba = torch.softmax(preds, 1) + eps
        preds_proba = preds_proba / preds_proba.sum(1, keepdim=True)
        super().update(preds_proba, target)


def mulc_vbwva_experiment(
    experiment_cfg_path: str = "data/config/mulc_vbwva.experiment.json",
):
    cfg, run_id, save_dir, logger = setup_experiment(experiment_cfg_path)

    experiment_name = cfg["name"]
    training_args = cfg["training"]
    model_args = cfg["model"]["args"]
    optimizer_args = cfg["optimizer"]["args"]
    dataset_cfgs = cfg["datasets"]
    trainer_args = dict(
        default_root_dir=save_dir,
        deterministic=True,
        accelerator="auto",
        max_epochs=training_args["epochs"],
        logger=logger,
    )

    lightning.seed_everything(training_args["seed"])

    ds_cfg = dataset_cfgs["train"]

    patch_size = model_args["input_dimensions"][0]
    cache_size = cfg.get("cache_size", 0)
    root_dir = Path(ds_cfg["base_dir"])
    df = pd.read_csv(root_dir / ds_cfg["path_df"])

    max_imgs = ds_cfg.get("max_images")
    if max_imgs is not None:
        df = df.sample(n=max_imgs)

    img_fps, mask_fps = df[["image_filepath", "mask_filepath"]].values.T

    dataset = PatchDataset(
        root_dir,
        img_fps,
        mask_fps,
        patch_size=patch_size,
        pad_value=ds_cfg["na_value"],
    )

    n = len(dataset)
    idxs = np.arange(n)
    np.random.shuffle(idxs)
    n_train = int(n * training_args["train_percentage"])
    idxs_train = idxs[:n_train]
    idxs_val = idxs[n_train:]

    dataloader_train = DataLoader(
        dataset=dataset,
        batch_size=training_args["batch_size"],
        sampler=torch.utils.data.sampler.SubsetRandomSampler(idxs_train),
        num_workers=training_args.get("num_workers"),
        pin_memory=True,
    )
    dataloader_val = DataLoader(
        dataset=dataset,
        batch_size=training_args["batch_size"],
        sampler=torch.utils.data.sampler.SubsetRandomSampler(idxs_val),
        num_workers=training_args.get("num_workers"),
        pin_memory=True,
    )

    meta_data = {
        "experiment_name": experiment_name,
        "run_id": run_id,
        "seed": training_args["seed"],
        "datasets": ds_cfg["name"],
    }

    def loss_fn(y_pred, y_true):
        kld = torch.nn.KLDivLoss(reduction="batchmean")
        softmax = torch.nn.LogSoftmax(dim=1)
        return kld(softmax(y_pred), y_true)

    ce = torch.nn.CrossEntropyLoss()

    metrics = {
        "train": {
            "cosine": CustomCosineSimilarity(reduction="mean"),
            "mse": torchmetrics.MeanSquaredError(),
        },
        "test": {
            "cosine": CustomCosineSimilarity(reduction="mean"),
            "kld": CustomKLDivergence(reduction="mean"),
            "mse": torchmetrics.MeanSquaredError(),
        },
    }

    run_train_test(
        dataloader_train,
        model_args,
        trainer_args,
        optimizer_args,
        metrics,
        meta_data,
        dataloader_val,
        None,
        criterion=ce,
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

    dataset = _load_and_preprocess_dataset(train_dataset_cfg)
    model_args["input_dimensions"] = dataset["input_dimensions"]
    model_args["num_classes"] = dataset["num_labels"]

    metrics = _get_metrics(dataset["num_labels"])

    meta_data = {
        "experiment_name": experiment_name,
        "run_id": run_id,
        "seed": training_args["seed"],
        "batch_size": training_args["batch_size"],
        "n_folds": training_args.get("n_folds", 5),
        "datasets": {
            "train": dataset["name"],
            "validation": dataset["name"],
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
        dataset["train"],
        model_args,
        trainer_args,
        optimizer_args,
        metrics,
        meta_data,
        dataset["test"],
    )


def ood_experiment(
    experiment_cfg_path: str = "./data/config/cross_location.experiment.json",
):
    cfg, run_id, save_dir, logger = setup_experiment(experiment_cfg_path)

    experiment_name = cfg["name"]
    training_args = cfg["training"]
    model_args = cfg["model"]["args"]
    optimizer_args = cfg["optimizer"]["args"]
    dataset_cfgs = cfg["datasets"]
    trainer_args = dict(
        default_root_dir=save_dir,
        deterministic=True,
        accelerator="auto",
        max_epochs=training_args["epochs"],
        logger=logger,
    )

    lightning.seed_everything(training_args["seed"])

    datasets = []
    for dataset_cfg in dataset_cfgs:
        datasets.append(load_benchmark_dataset(**dataset_cfg["files"]))

    batch_size = training_args["batch_size"]
    for i in range(len(datasets)):
        test_sets = datasets.copy()
        train_set = test_sets.pop(i)

        features, labels = train_set

        features_to_process = ["data_HS_LR"]
        X, y, (train_idxs, test_idxs), label_encoder, pcas = preprocess(features, labels, features_to_process=features_to_process)

        num_labels = len(label_encoder.classes_)
        model_args["input_dimensions"] = X.shape[2:]
        model_args["num_classes"] = num_labels
        metrics = _get_metrics(num_labels)

        train_dataset = torch.utils.data.TensorDataset(
            torch.tensor(X[train_idxs]),
            torch.tensor(y[train_idxs])
        )
        test_dataset = torch.utils.data.TensorDataset(
            torch.tensor(X[test_idxs]),
            torch.tensor(y[test_idxs])
        )

        n = len(train_dataset)
        idxs = np.random.permutation(n)
        n_train = int(n * training_args["train_percentage"])
        idxs_train = idxs[:n_train]
        idxs_val = idxs[n_train:]

        test_datasets = [test_dataset]
        for features, labels in test_sets:
            X, y, (train_idxs, test_idxs), _, _ = preprocess(
                features,
                labels,
                features_to_process=features_to_process,
                pcas=pcas,
                label_encoder=label_encoder,
            )
            idxs = np.concatenate([train_idxs[0], test_idxs[0]])
            test_datasets.append(torch.utils.data.TensorDataset(
                torch.tensor(X[idxs]),
                torch.tensor(y[idxs]),
            ))

        train_dataloader = _get_dataloader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            idxs=idxs_train,
        )
        val_dataloader = _get_dataloader(
            train_dataset,
            batch_size=batch_size,
            shuffle=False,
            idxs=idxs_val,
        )
        test_dataloader = [
            _get_dataloader(
                test_ds,
                batch_size=batch_size,
                shuffle=False,
                num_workers=min(multiprocessing.cpu_count(), 6),
            )
            for test_ds in test_datasets
        ]

        meta_data = {
            "experiment_name": experiment_name,
            "run_id": run_id,
            "seed": training_args["seed"],
            "datasets": " | ".join([ds["name"] for ds in datasets]),
        }

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
