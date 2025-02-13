from typing import Literal

import torch
import torchmetrics


class StableProbabilityMixin:
    def preprocess_inputs(self, preds, target, epsilon=1e-8):
        target = torch.nan_to_num(target, 0.0)
        target = target + epsilon
        target = target / target.sum(1, keepdim=True)

        preds_proba = torch.softmax(preds, 1) + epsilon
        preds_proba = preds_proba / preds_proba.sum(1, keepdim=True)

        return preds_proba, target


class StableCosineSimilarity(StableProbabilityMixin, torchmetrics.CosineSimilarity):
    def __init__(self, epsilon=1e-8, **kwargs):
        super().__init__(**kwargs)
        self.epsilon = epsilon

    def update(self, preds, target):
        preds, target = self.preprocess_inputs(preds, target, self.epsilon)
        super().update(preds, target)


class StableKLDivergence(StableProbabilityMixin, torchmetrics.KLDivergence):
    def __init__(self, epsilon=1e-8, **kwargs):
        super().__init__(**kwargs)
        self.log_prob = False
        self.epsilon = epsilon

    def update(self, preds, target):
        preds, target = self.preprocess_inputs(preds, target, self.epsilon)
        super().update(preds, target)


def js_divergence(preds, target, log_prob=False, reduction: Literal["mean", "sum"]="mean", normalize=True):
    m = 0.5 * (preds + target)
    kld_p = torchmetrics.functional.kl_divergence(preds, m, log_prob=log_prob, reduction=reduction)
    kld_q = torchmetrics.functional.kl_divergence(target, m, log_prob=log_prob, reduction=reduction)
    jsd = (kld_p + kld_q) * 0.5

    if normalize:
        jsd = jsd / torch.log(torch.tensor(2.0))

    return jsd

class StableJensonShannonDivergence(StableProbabilityMixin, torchmetrics.Metric):
    def __init__(self, epsilon=1e-8, reduction: Literal["mean", "sum"]="mean", normalize=True, **kwargs):
        super().__init__(**kwargs)
        self.reduction = reduction
        self.epsilon = epsilon
        self.normalize = normalize
        self.add_state("measures", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0, dtype=torch.int64), dist_reduce_fx="sum")

    def update(self, preds, target):
        preds, target = self.preprocess_inputs(preds, target, self.epsilon)
        self.measures += js_divergence(preds, target, False, "sum", self.normalize)
        self.total += target.shape[0]

    def compute(self):
        if self.reduction == "mean":
            return self.measures / self.total
        else:
            return self.measures


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


def default_classification_metrics(num_classes):
    task = "binary" if num_classes == 2 else "multiclass"
    return dict(
        train=dict(
            accuracy=torchmetrics.Accuracy(task=task, num_classes=num_classes),
        ),
        test=dict(
            accuracy=torchmetrics.Accuracy(task=task, num_classes=num_classes),
        ),
    )
