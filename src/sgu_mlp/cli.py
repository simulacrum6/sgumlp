import datetime
from pathlib import Path
import multiprocessing

import gdown
import lightning
import numpy as np
import patoolib
import sklearn.model_selection
import torch
import torchmetrics
from sklearn.model_selection import KFold

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
    return torch.utils.data.TensorDataset(
        torch.from_numpy(X), torch.from_numpy(y)
    )

def download_datasets():
    data_path = Path('data')
    data_path.mkdir(parents=True, exist_ok=True)
    file_id = "1dLJJrNJpQoQeDHybs37iGxmrSU6aP2xv" # https://drive.usercontent.google.com/download?id=1dLJJrNJpQoQeDHybs37iGxmrSU6aP2xv&export=download
    file_path = data_path / (file_id + '.rar')
    try:
        gdown.download(id=file_id, output=str(file_path), quiet=False)
        patoolib.extract_archive(str(file_path), outdir=str(data_path))
    finally:
        file_path.unlink(missing_ok=True)

def reproduction():
    # TODO: use config
    # training parameters
    n_epochs = 100
    n_folds = 5
    batch_size = 256
    seed = 271828182
    lightning.seed_everything(seed)

    # experiment parameters
    run_id = f"reproduction__{int(datetime.datetime.now().timestamp())}"
    cfg_path = Path("data/config/augsburg.dataset.json")
    save_dir = Path(f"data/runs/{run_id}")
    model_dir = save_dir / "checkpoints"

    # load data
    dataset = Dataset.from_json(cfg_path)
    X_train, X_test, y_train, y_test = preprocess(dataset)
    dataset_train = torch.utils.data.TensorDataset(
        torch.from_numpy(X_train), torch.from_numpy(y_train)
    )
    dataset_test = torch.utils.data.TensorDataset(
        torch.from_numpy(X_test), torch.from_numpy(y_test)
    )

    model_args = dict(
        input_dimensions=X_train.shape[1:],
        token_features=256,
        mixer_features_channel=256,
        mixer_features_sequence=256,
        dwc_kernels=(1, 3, 5),
        num_blocks=1,
        activation="gelu",
        residual_weight=2,
        learnable_residual=False,
        embedding_kernel_size=4,
        num_classes=dataset.num_labels,
        dropout=0.4,
    )

    optimizer_args = dict(lr=0.001, weight_decay=0.0001)

    # generate metrics
    task = "binary" if dataset.num_labels == 2 else "multiclass"
    metric_args = dict(task=task, num_classes=dataset.num_labels)
    metrics = dict(
        train=dict(
            accuracy=torchmetrics.Accuracy(**metric_args),
        ),
        test=dict(
            accuracy=torchmetrics.Accuracy(**metric_args),
            precision=torchmetrics.Precision(**metric_args),
            recall=torchmetrics.Recall(**metric_args),
            f1=torchmetrics.F1Score(**metric_args),
        )
    )

    test_dataloader = _get_dataloader(dataset_test, batch_size, None, shuffle=False)
    cv = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    for fold, (train_idx, val_idx) in enumerate(cv.split(dataset_train)):
        meta_data = {
            'run_id': run_id,
            'seed': seed,
            'cv': {
                'fold': fold,
                'n_folds': n_folds,
            },
            'datasets': {
                'train': dataset.name,
                'validation': dataset.name,
            }
        }
        train_dataloader = _get_dataloader(dataset_train, batch_size, train_idx, shuffle=True)
        val_dataloader = _get_dataloader(dataset_train, batch_size, val_idx, shuffle=False)
        model = LitSGUMLPMixer(model_args, optimizer_args, metrics, meta_data=meta_data)
        checkpoint_cb = lightning.pytorch.callbacks.ModelCheckpoint(
            filename=f"{fold}" + "_{epoch}-{step}",
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

def cross_location():
    # training parameters
    n_epochs = 100
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
        )
    )

    # experiment parameters
    run_id = f"cross_location__{int(datetime.datetime.now().timestamp())}"
    save_dir = Path(f"data/runs/{run_id}")
    model_dir = save_dir / "checkpoints"

    # load data
    # TODO: fixme
    augsburg = Dataset.from_json(Path("data/config/augsburg.dataset.json"))
    berlin = Dataset.from_json(Path("data/config/augsburg.dataset.json"))

    xy_augsburg = preprocess(augsburg, return_train_test=False)
    xy_berlin = preprocess(berlin, return_train_test=False)

    dataset_pairs = [
        (xy_augsburg, xy_berlin),
        (xy_berlin, xy_augsburg),
    ]

    for i, ((X, y), (X_test, y_test)) in enumerate(dataset_pairs):
        train_percentage = 1 / n_folds
        X_train, X_val, y_train, y_val = sklearn.model_selection.train_test_split(X, y, train_size=train_percentage, random_state=seed)

        train_dataloader = _get_dataloader(_to_torch_dataset(X_train, y_train), batch_size=batch_size, shuffle=True, num_workers=multiprocessing.cpu_count())
        val_dataloader = _get_dataloader(_to_torch_dataset(X_val, y_val), batch_size=batch_size, shuffle=False, num_workers=multiprocessing.cpu_count())
        test_dataloader = _get_dataloader(_to_torch_dataset(X_test, y_test), batch_size=batch_size, shuffle=False, num_workers=multiprocessing.cpu_count())

        meta_data = {
            'run_id': run_id,
            'seed': seed,
            'datasets': {
                'train': i,
                'validation': i,
                'test': (i + 1) % len(dataset_pairs),
            }
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