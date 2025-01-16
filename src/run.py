import datetime
import multiprocessing
from pathlib import Path

import lightning
import torch
import torchmetrics
from sklearn.model_selection import KFold

from data import Dataset, preprocess
from models import LitSGUMLPMixer


def get_dataloader(dataset, batch_size, idxs=None):
    sampler = None
    if idxs is not None:
        sampler = torch.utils.data.SubsetRandomSampler(idxs)
    return torch.utils.data.DataLoader(
        dataset,
        num_workers=1,
        batch_size=batch_size,
        sampler=sampler,
    )


def reproduction():
    seed = 271828182
    lightning.seed_everything(seed)

    # experiment parameters
    run_id = f"{int(datetime.datetime.now().timestamp())}"
    cfg_path = Path("data/.augsburg.json")
    save_dir = Path(f"data/runs/{run_id}")
    model_dir = save_dir / "checkpoints"

    # training parameters
    n_epochs = 100
    n_folds = 5
    batch_size = 256

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

    test_dataloader = get_dataloader(dataset_test, batch_size, None)
    cv = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    for fold, (train_idx, val_idx) in enumerate(cv.split(dataset_train)):
        train_dataloader = get_dataloader(dataset_train, batch_size, train_idx)
        val_dataloader = get_dataloader(dataset_train, batch_size, val_idx)
        model = LitSGUMLPMixer(model_args, optimizer_args, metrics)
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


if __name__ == "__main__":
    reproduction()
