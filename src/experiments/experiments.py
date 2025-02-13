import datetime
import json
import multiprocessing
from pathlib import Path

import lightning
import numpy as np
import pandas as pd
import requests
import torch
import torchmetrics
from torch.utils.data import DataLoader

from .data import load_benchmark_dataset, preprocess, PatchDataset, get_dataloader, load_and_preprocess_dataset
from .metrics import get_metrics, StableCosineSimilarity, StableKLDivergence
from .train import run_train_test, run_cv


def get_experiment_logger(experiment_name, run_id, tracking_uri: str | None, save_local: bool):
    if tracking_uri is not None and not save_local:
        try:
            requests.get(tracking_uri + "/health")
        except Exception:
            print("Tracking uri doesn't exist, saving locally to ./ml-runs")
            tracking_uri = None

    return lightning.pytorch.loggers.MLFlowLogger(
        experiment_name=experiment_name, run_name=run_id, tracking_uri=tracking_uri
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

    logger = get_experiment_logger(
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
        deterministic=False,
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

    ce = torch.nn.CrossEntropyLoss()

    metrics = {
        "train": {
            "cosine": StableCosineSimilarity(reduction="mean"),
            "mse": torchmetrics.MeanSquaredError(),
        },
        "test": {
            "cosine": StableCosineSimilarity(reduction="mean"),
            "kld": StableKLDivergence(reduction="mean"),
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

    dataset = load_and_preprocess_dataset(train_dataset_cfg)
    model_args["input_dimensions"] = dataset["input_dimensions"]
    model_args["num_classes"] = dataset["num_labels"]

    metrics = get_metrics(dataset["num_labels"])

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
        deterministic=False,
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
        deterministic=False,
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
        metrics = get_metrics(num_labels)

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

        train_dataloader = get_dataloader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            idxs=idxs_train,
        )
        val_dataloader = get_dataloader(
            train_dataset,
            batch_size=batch_size,
            shuffle=False,
            idxs=idxs_val,
        )
        test_dataloader = [
            get_dataloader(
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
