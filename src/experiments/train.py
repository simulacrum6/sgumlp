import lightning
import torch
from sklearn.model_selection import KFold

from .data import get_dataloader
from .models import LitSGUMLPMixer


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
        get_dataloader(dataset_test, batch_size, None, shuffle=False)
        if dataset_test
        else None
    )

    for fold, (train_idx, val_idx) in enumerate(cv.split(dataset_train)):
        train_dataloader = get_dataloader(
            dataset_train, batch_size, train_idx, shuffle=True
        )
        val_dataloader = get_dataloader(
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
