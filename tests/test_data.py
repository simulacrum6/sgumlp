import numpy as np

from sgu_mlp.data import patchify, reduce_dimensions, load_benchmark_dataset, preprocess


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


def test_load_benchmark_dataset(benchmark_dataset_info):
    base_dir, feature_files, label_files, na_value = benchmark_dataset_info

    features, labels = load_benchmark_dataset(
        base_dir, feature_files, label_files, na_value=na_value
    )
    assert len(features) == len(feature_files)
    assert len(labels) == len(label_files)

    label_files = [label_files[0], None]
    features, labels = load_benchmark_dataset(
        base_dir, feature_files, label_files, na_value=na_value
    )
    assert len(features) == len(feature_files)
    assert len(labels) == len(label_files)

    train_labels, test_labels = labels.values()
    assert train_labels.shape == test_labels.shape
    assert np.all(test_labels == na_value)


def test_preprocess(benchmark_dataset):
    features, labels, na_value = benchmark_dataset

    c = sum(feat.shape[-1] for feat in features.values())
    h, w, k = labels["train"].shape
    n = h * w
    p = 11

    X, y, idxs, label_encoder, pcas = preprocess(
        features, labels, patch_size=p, na_value=na_value, features_to_process=None
    )
    assert y.shape == (n,)
    assert X.shape == (n, c, p, p)
    for idx in idxs:
        assert X[idx].shape[0] == y[idx].shape[0]
        assert X[idx].shape[1:] == X.shape[1:]

    assert pcas == {}
    assert label_encoder is not None

    num_components = 2
    name, feat = next(iter(features.items()))
    features_to_process = [name]
    c_ = c - feat.shape[-1] + num_components
    X, y, idxs, label_encoder, pcas = preprocess(
        features,
        labels,
        patch_size=p,
        na_value=na_value,
        features_to_process=features_to_process,
        num_components=num_components,
    )

    assert name in pcas
    assert X.shape[1] == c_
    X_, _, _, _, _ = preprocess(
        features,
        labels,
        patch_size=p,
        na_value=na_value,
        num_components=num_components,
        features_to_process=features_to_process,
        pcas=pcas,
    )

    assert np.all(X == X_)
