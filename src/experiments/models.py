import lightning
import torch
import torchmetrics

from experiments.metrics import default_classification_metrics
from sgu_mlp import SGUMLPMixer


class LitSGUMLPMixer(lightning.LightningModule):
    def __init__(
        self,
        model_params,
        optimizer_params,
        metrics=None,
        criterion=None,
        *args,
        **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["metrics", "criterion"])

        if metrics is None:
            metrics = default_classification_metrics(self.hparams.model_params["num_classes"])
        self.model = SGUMLPMixer(**model_params)
        if criterion is None:
            self.criterion = torch.nn.CrossEntropyLoss()
        else:
            self.criterion = criterion
        self.optimizer_cls = torch.optim.AdamW

        self.train_metrics = torchmetrics.MetricCollection(
            dict(
                metrics.get("train", default_classification_metrics(self.model.num_classes)["train"])
            )
        )
        self.test_metrics = torchmetrics.MetricCollection(
            dict(metrics.get("test", default_classification_metrics(self.model.num_classes)["test"]))
        )

    def forward(self, x):
        return self.model(x)

    def _step(self, batch, batch_idx, split, on_step=False, on_epoch=True):
        x, y = batch
        y_hat = self.forward(x)
        loss = self.criterion(y_hat, y)

        self.log(f"{split}_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        metrics = self.train_metrics if split == "train" else self.test_metrics
        for name, metric in metrics.items():
            self.log(
                f"{split}_{name}",
                metric(y_hat, y),
                on_step=on_step,
                on_epoch=on_epoch,
                prog_bar=True,
            )
        return loss

    def training_step(self, batch, batch_idx):
        return self._step(batch, batch_idx, "train", on_step=True, on_epoch=False)

    def validation_step(self, batch, batch_idx):
        with torch.no_grad():
            return self._step(batch, batch_idx, "val", on_step=False, on_epoch=True)

    def test_step(self, batch, batch_idx):
        with torch.no_grad():
            return self._step(batch, batch_idx, "test", on_step=False, on_epoch=True)

    def configure_optimizers(self):
        return self.optimizer_cls(
            self.model.parameters(), **self.hparams.optimizer_params
        )
