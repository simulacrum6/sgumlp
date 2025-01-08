from src.data import patchify
import numpy as np


def test_patchify(image_and_labels):
    image, labels = image_and_labels
    patch_size = 11

    X, y = patchify(image, labels, patch_size=patch_size)
    assert X.shape[0] == y.shape[0]
    assert X.shape[1:3] == (patch_size, patch_size)
    assert X.shape[-1] == image.shape[-1]

    X, y = patchify(image, labels, patch_size=patch_size, only_valid=False)
    assert X.shape[0] == np.prod(image.shape[:2])

    X, y = patchify(image)
    assert X.shape[0] == np.prod(image.shape[:2])
    assert y is None
