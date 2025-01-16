from functools import partial

import lightning
import sklearn
from sklearn.model_selection import KFold

from data import Dataset, preprocess
from models import SGUMLPMixer, LitSGUMLPMixer
import torch
import numpy as np
import random
from pathlib import Path
import datetime
import pandas as pd


def get_dataloader(dataset, batch_size, idxs=None):
    sampler = None
    if idxs is not None:
        sampler = torch.utils.data.SubsetRandomSampler(idxs)
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
    )


def reproduction_lit():
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

    cv = KFold(n_splits=n_folds, shuffle=True, random_state=seed)

    average = "micro"
    metrics = [
        ("accuracy", sklearn.metrics.accuracy_score),
        ("balanced_accuracy", sklearn.metrics.balanced_accuracy_score),
        ("f1_micro", partial(sklearn.metrics.f1_score, average=average)),
        ("precision_micro", partial(sklearn.metrics.precision_score, average=average)),
        ("recall_micro", partial(sklearn.metrics.precision_score, average=average)),
    ]

    test_dataloader = get_dataloader(dataset_test, batch_size, None)

    for fold, (train_idx, val_idx) in enumerate(cv.split(dataset_train)):
        train_dataloader = get_dataloader(dataset_train, batch_size, train_idx)
        val_dataloader = get_dataloader(dataset_train, batch_size, val_idx)
        model = LitSGUMLPMixer(model_args, optimizer_args)
        checkpoint_cb = lightning.pytorch.callbacks.ModelCheckpoint(
            filename=f"{fold}" + "_{epoch}-{step}"
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


def reproduction():
    """
    Runs model training, aiming to replicate the results from the original study.
    """
    seed = 271828182
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    run_id = f"{int(datetime.datetime.now().timestamp())}"
    cfg_path = Path("data/.augsburg.json")
    save_dir = Path(f"data/runs/{run_id}")
    model_dir = save_dir / "models"
    results_dir = save_dir / "results"
    for dir in [save_dir, model_dir, results_dir]:
        dir.mkdir(parents=True, exist_ok=True)

    dataset = Dataset.from_json(cfg_path)
    X_train, X_test, y_train, y_test = preprocess(dataset)
    dataset_train = torch.utils.data.TensorDataset(
        torch.from_numpy(X_train), torch.from_numpy(y_train)
    )
    dataset_test = torch.utils.data.TensorDataset(
        torch.from_numpy(X_test), torch.from_numpy(y_test)
    )

    n_classes = dataset.num_labels
    n_epochs = 100
    n_folds = 5
    weight_decay = 0.0001
    batch_size = 256
    learning_rate = 0.001

    cv = KFold(n_splits=n_folds, shuffle=True)
    dropout_rate = 0.4

    criterion = torch.nn.CrossEntropyLoss()
    average = "micro"
    metrics = [
        ("accuracy", sklearn.metrics.accuracy_score),
        ("balanced_accuracy", sklearn.metrics.balanced_accuracy_score),
        ("f1_micro", partial(sklearn.metrics.f1_score, average=average)),
        ("precision_micro", partial(sklearn.metrics.precision_score, average=average)),
        ("recall_micro", partial(sklearn.metrics.precision_score, average=average)),
    ]

    def prepare_model_and_optimizer():
        model = SGUMLPMixer(
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
            num_classes=n_classes,
            dropout=dropout_rate,
        )
        model.to(device)
        optimizer = torch.optim.Adam(
            model.parameters(), lr=learning_rate, weight_decay=weight_decay
        )
        return model, optimizer

    def get_dataloaders(train_idx, val_idx):
        train_loader = torch.utils.data.DataLoader(
            dataset_train,
            batch_size=batch_size,
            sampler=torch.utils.data.sampler.SubsetRandomSampler(train_idx),
        )
        val_loader = torch.utils.data.DataLoader(
            dataset_train,
            batch_size=batch_size,
            sampler=torch.utils.data.sampler.SubsetRandomSampler(val_idx),
        )
        return train_loader, val_loader

    def train_epoch(model, optimizer, dataloader, criterion, device):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            n_samples = inputs.shape[0]

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += n_samples
            correct += predicted.eq(targets).sum().item()

        epoch_loss = running_loss / len(dataloader)
        epoch_acc = 100.0 * correct / total
        return epoch_loss, epoch_acc

    def evaluate(model, dataloader, criterion, device, metrics=None):
        if metrics is None:
            metrics = [("accuracy_balanced", sklearn.metrics.balanced_accuracy_score)]

        model.eval()
        running_loss = 0.0
        total = 0
        correct = 0
        y_pred = []
        y_true = []

        with torch.no_grad():
            for inputs, targets in dataloader:
                inputs, targets = inputs.to(device), targets.to(device)
                n_samples = inputs.shape[0]

                outputs = model(inputs)
                loss = criterion(outputs, targets)

                running_loss += loss.item()
                _, predicted = outputs.max(1)
                total += n_samples
                correct += predicted.eq(targets).sum().item()

                y_pred.extend(predicted.ravel().tolist())
                y_true.extend(targets.ravel().tolist())

        val_loss = running_loss / len(dataloader)
        val_acc = 100.0 * correct / total
        scores = [(name, metric(y_true, y_pred)) for name, metric in metrics]

        print(sklearn.metrics.classification_report(y_true, y_pred))
        return val_loss, val_acc, scores, (y_true, y_pred)

    def to_records(scores, run_id, fold, epoch, split):
        return [
            dict(
                metric=name,
                score=score,
                run_id=run_id,
                fold=fold,
                epoch=epoch,
                split=split,
            )
            for name, score in scores
        ]

    print("Starting Training")
    results = []
    for fold, (train_idx, val_idx) in enumerate(cv.split(X_train)):
        print(f"### Fold {fold + 1}/{n_folds}")
        model, optimizer = prepare_model_and_optimizer()
        dataloader_train, dataloader_val = get_dataloaders(train_idx, val_idx)

        for epoch in range(n_epochs):
            epoch_loss, epoch_acc = train_epoch(
                model, optimizer, dataloader_train, criterion, device
            )
            epoch_val_loss, epoch_val_acc, epoch_scores, _ = evaluate(
                model, dataloader_val, criterion, metrics=metrics, device=device
            )
            epoch_scores.append(("loss", epoch_loss))
            results.extend(to_records(epoch_scores, run_id, fold, epoch, "validation"))

        torch.save(
            {
                "fold": fold,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "run_id": run_id,
            },
            model_dir / f"model_{run_id}_{fold}.pt",
        )

        dataloader_test = torch.utils.data.DataLoader(
            dataset_test, batch_size=batch_size, shuffle=False
        )
        test_loss, test_acc, test_scores, _ = evaluate(
            model, dataloader_test, criterion, metrics=metrics, device=device
        )
        test_scores.append(("loss", test_loss))
        results.extend(to_records(test_scores, run_id, fold, -1, "test"))

    pd.DataFrame(results).to_csv(results_dir / "metrics.csv", index=False)


if __name__ == "__main__":
    reproduction_lit()
