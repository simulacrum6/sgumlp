import torch
import torchmetrics


def get_metrics(num_labels: int):
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
