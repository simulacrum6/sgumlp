from sgu_mlp.data import patchify


def test_patchify(image_and_labels):
    image, labels = image_and_labels
    patch_size = 11
    h, w, c = image.shape

    X = patchify(image, patch_size=patch_size)
    assert X.shape[:2] == (h, w)
    assert X.shape[2:4] == (patch_size, patch_size)
    assert X.shape[-1] == c
