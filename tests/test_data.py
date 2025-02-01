import numpy as np

from sgu_mlp.data import patchify, reduce_dimensions


def test_patchify(image_and_labels):
    image, labels = image_and_labels
    patch_size = 11
    h, w, c = image.shape

    X = patchify(image, patch_size=patch_size)
    assert X.shape[:2] == (h, w)
    assert X.shape[2:4] == (patch_size, patch_size)
    assert X.shape[-1] == c


def test_reduce_dimensions(hs_image):
    hs_image = hs_image.numpy()
    b, c, h, w = hs_image.shape
    image = hs_image[0]
    num_components = 4

    scaled_img, pca = reduce_dimensions(
        image, num_components=num_components, channel_dim=0, train_mask=None, pca=None
    )
    assert scaled_img.shape == (num_components, h, w)

    scaled_img_2, _ = reduce_dimensions(
        image, num_components=num_components, channel_dim=0, train_mask=None, pca=pca
    )
    assert np.all(scaled_img_2 == scaled_img)

    channel_last_img = np.swapaxes(scaled_img, 0, -1)
    scaled_img, _ = reduce_dimensions(
        channel_last_img, num_components=num_components, channel_dim=-1, train_mask=None
    )
    assert scaled_img.shape == (h, w, num_components)

    train_indices = np.random.randint(0, 2, (h, w)).astype(bool)
    scaled_img, _ = reduce_dimensions(
        image,
        num_components=num_components,
        channel_dim=0,
        train_mask=train_indices,
        pca=None,
    )
    assert scaled_img.shape == (num_components, h, w)
    assert ~np.all(scaled_img_2 == scaled_img)

    train_indices = np.where(np.random.randint(0, 2, (h, w)).astype(bool))
    scaled_img, _ = reduce_dimensions(
        channel_last_img,
        num_components=num_components,
        channel_dim=-1,
        train_mask=train_indices,
        pca=None,
    )
    assert scaled_img.shape == (h, w, num_components)

    scaled_imgs, pca = reduce_dimensions(
        hs_image,
        num_components=num_components,
        channel_dim=1,
        train_mask=None,
        pca=None,
    )
    assert scaled_imgs.shape == (b, num_components, h, w)

    train_indices = np.random.randint(0, 2, (b, h, w)).astype(bool)
    scaled_imgs_2, _ = reduce_dimensions(
        hs_image,
        num_components=num_components,
        channel_dim=1,
        train_mask=train_indices,
        pca=None,
    )
    assert scaled_imgs_2.shape == (b, num_components, h, w)
    assert ~np.all(scaled_imgs_2 == scaled_imgs)
