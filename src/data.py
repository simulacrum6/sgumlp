import json
from dataclasses import dataclass
from pathlib import Path
import scipy.io as sio
import numpy as np

from numpy.lib.stride_tricks import sliding_window_view
from sklearn.decomposition import PCA


def patchify(
    image: np.ndarray,
    labels=None,
    patch_size: int = 9,
    pad_value=0,
    mask_value=0,
    only_valid=True,
):
    """Extract square patches centered on each pixel of a 2D or N-D image array.

    Args:
        image: Input array of shape (H, W, ...) where H and W are spatial dimensions.
            Additional dimensions after the first two are preserved as features.
        labels: Label array of shape (H, W). When provided with only_valid=True, patches
            will only be extracted for pixels that have labels (where labels != mask_value).
            Defaults to None.
        patch_size: Width and height of patches. Must be odd to ensure patches are centered.
            Defaults to 9.
        pad_value: Value used to pad borders of image before extracting patches.
            Defaults to 0.
        mask_value: Value in labels array that indicates pixels without a valid label.
            Defaults to 0.
        only_valid: If True and labels is provided, only extract patches centered on labeled
            pixels (where labels != mask_value). If False or labels=None, extract patches
            for all pixels. Defaults to True.

    Returns:
        A tuple of (patches, labels):
            patches: Extracted patches with shape:
                - (N, patch_size, patch_size, ...) if only_valid=True and labels provided,
                  where N is number of labeled pixels
                - (H*W, patch_size, patch_size, ...) otherwise,
                  where H*W is total number of pixels
            labels: If labels provided, returns corresponding label for each patch:
                - Shape (N,) if only_valid=True
                - Shape (H*W,) otherwise
                If labels=None, returns None

    Raises:
        ValueError: If
            - patch_size is even
            - image has fewer than 2 dimensions
            - spatial dimensions of image and labels do not match.

    Notes:
        - Patches are extracted using sliding_window_view after padding
        - Each patch is centered on its corresponding pixel
        - Feature dimensions (after H,W) are preserved in output patches
    """
    if patch_size % 2 == 0:
        raise ValueError("patch_size must be odd")
    offset = patch_size // 2

    image = np.array(image, copy=True)
    labels = np.array(labels, copy=True) if labels is not None else None

    if image.ndim < 2:
        raise ValueError("image must have at least 2 dimensions")

    if (labels is not None) and (image.shape[:2] != labels.shape[:2]):
        raise ValueError(
            "first two dimensions of image and labels must have the same shape"
        )

    # configure sliding windows
    pad_width = [(offset, offset), (offset, offset)]
    window_shape = [patch_size, patch_size]

    # include feature dimensions
    if image.ndim > 2:
        n_feature_dims = image.ndim - 2
        pad_width.extend([(0, 0)] * n_feature_dims)
        window_shape.extend(image.shape[2:])

    # extract patches
    image = np.pad(
        image, pad_width=pad_width, mode="constant", constant_values=pad_value
    )
    image = sliding_window_view(image, window_shape=window_shape).squeeze()

    # filter for valid pixels and reshape
    if (labels is not None) and only_valid:
        i, j = np.where(labels != mask_value)
        image = image[i, j, ...]
        labels = labels[i, j]
    else:
        image = image.reshape(-1, *image.shape[2:])

    return image, labels


@dataclass
class DatasetConfig:
    name: str
    base_dir: str
    feature_files: list[str]
    labels_file: str
    labels_file_test: str | None = None

    @classmethod
    def from_json(cls, path):
        with open(path) as f:
            return cls(**json.load(f))


class Dataset:
    def __init__(self, name: str, features, labels_train, labels_test=None):
        self.name = name
        self._features = features
        self._labels_train = labels_train
        self._labels_test = labels_test

    def data(self, features: list[str] | None = None):
        if features is None:
            feats = [f["data"] for _, f in self._features.items()]
        else:
            feats = [self._features[f]["data"] for f in features]

        return np.concatenate(feats, axis=-1)

    def feature(self, name):
        return self._features[name]

def _load_matrix(fp: Path):
    fp = Path(fp)
    return fp.stem, sio.loadmat(str(fp))[fp.stem]

def load_dataset(cfg: DatasetConfig):
    base_path = Path(cfg.base_dir)

    features = []
    feature_names = []
    sizes = []
    for file_name in cfg.feature_files:
        file_path = base_path / file_name
        feature_name, matrix = _load_matrix(file_path)
        if matrix.ndim == 2:
            matrix = np.expand_dims(matrix, axis=-1)

        size = matrix.shape[-1]

        features.append(matrix)
        feature_names.append(feature_name)
        sizes.append(size)

    image = np.concatenate(features, axis=-1)
    feature_info = {
        feature_name: {"name": feature_name, "size": size, "data": data}
        for feature_name, size, data in zip(feature_names, sizes, features)
    }

    _, labels = _load_matrix(base_path / cfg.labels_file)

    if cfg.labels_file_test is not None:
        _, labels_test = _load_matrix(base_path / cfg.labels_file_test)
    else:
        labels_test = None

    return {
        "name": cfg.name,
        "features": feature_info,
        "dimensions": image.shape,
        "data": image,
        "labels": labels,
        "labels_test": labels_test if labels_test is not None else None,
    }


def reduce_dimensions(data, num_components=15, valid_indices=None, return_pca=False):
    X = data
    h, w, c = X.shape

    if valid_indices is not None:
        X = X[valid_indices]

    if X.ndim > 2:
        X = X.reshape(-1, c)

    pca = PCA(n_components=num_components, whiten=True).fit(X)
    X_reduced = pca.transform(data.reshape(-1, c)).reshape(h, w, num_components)

    if return_pca:
        return X_reduced, pca

    return X_reduced

def make_image(dataset, features: list[str] | None = None, missing_value=0):
    if features is None:
        features = dataset.features.keys()
    h, w = dataset["dimensions"][:2]

    X = []
    current_channel = 0
    for feature in features:
        feat = dataset["features"].get(feature)
        if feat is None:
            print(f'Feature "{feat}" not in dataset. Adding missing values ({missing_value} at channel {current_channel}).')
            X.append(np.full((h, w, 1), missing_value))
            current_channel += 1
        else:
            X.append(feat["data"])
            current_channel += feat["size"]

    return np.concatenate(X, axis=-1)


def preprocess(dataset):
    labels_train = dataset["labels"]
    labels_test = dataset["labels_test"]

    hs_feat = dataset["features"]["data_HS_LR"]
    hs = hs_feat["data"]
    n_components = 15
    train_indices = labels_train != 0

    hs_pca, pca = reduce_dimensions(hs, n_components, train_indices)
    name = 'data_HS_LR_pca15'
    dataset['features'][name] = {'name': name, 'size': n_components, 'data': hs_pca}

    features = [
        'data_DSM',
        'data_HS_LR_pca15',
        'data_SAR_HR',
    ]
    image = make_image(dataset, features)

    X_train, y_train = patchify(image, labels_train)
    X_test, y_test = patchify(image, labels_test)
    return X_train, X_test, y_train, y_test

def load_data_from_json_and_preprocess(json_path: str):
    cfg = DatasetConfig.from_json(json_path)
    dataset = load_dataset(cfg)
    X_train, X_test, y_train, y_test = preprocess(dataset)
    return X_train, X_test, y_train, y_test, dataset

if __name__ == "__main__":
    X_train, X_test, y_train, y_test, dataset = load_data_from_json_and_preprocess(json_path=".berlin.json")


