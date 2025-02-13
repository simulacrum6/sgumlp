import torch
from torchmetrics.functional import kl_divergence, cosine_similarity

from experiments.metrics import StableKLDivergence, StableJensonShannonDivergence, StableCosineSimilarity, js_divergence


def test_equivalence():
    reduction = "mean"
    preds = torch.randn(256, 8) * 10.0
    target = torch.softmax(torch.randn_like(preds), -1)

    cosine_stable = StableCosineSimilarity(reduction=reduction)
    cos_ref = cosine_similarity(torch.softmax(preds, -1), target, reduction=reduction)
    cos = cosine_stable(preds, target)
    assert torch.allclose(cos, cos_ref)

    kld_stable = StableKLDivergence(reduction=reduction)
    kld_ref = kl_divergence(torch.softmax(preds, -1), target, log_prob=False, reduction=reduction)
    kld = kld_stable(preds, target)
    assert torch.allclose(kld, kld_ref)

def test_jenson_shannon():
    reduction = "mean"

    p = torch.tensor([[0.0, 1.0]])
    q = torch.tensor([[1.0, 0.0]])

    assert torch.allclose(js_divergence(p, q, normalize=True), torch.tensor(1.0))
    assert torch.allclose(js_divergence(p, p, normalize=True), torch.tensor(0.0))

    preds = torch.randn(256, 8) * 10.0
    target = torch.softmax(torch.randn_like(preds), -1)
    target_correct = torch.softmax(preds, -1)
    target_incorrect = torch.softmax(preds * -1, -1)

    js_stable = StableJensonShannonDivergence(reduction=reduction)
    assert torch.allclose(js_stable(preds, target_correct), torch.tensor(0.0))

    js_stable.reset()
    assert torch.allclose(js_stable(preds, target_incorrect), torch.tensor(1.0), atol=1e-3)

    js_stable.reset()
    js_stable.update(preds, target)
    js_one = js_stable.compute()
    js_stable.update(preds, target_correct)
    js_two = js_stable.compute()

    assert torch.allclose(js_one / 2.0, js_two)
