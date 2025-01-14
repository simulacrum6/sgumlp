import json
from dataclasses import dataclass
from pathlib import Path
import scipy.io as sio
import numpy as np
import typing
from PIL import Image
from numpy.lib.stride_tricks import sliding_window_view
from sklearn.decomposition import PCA


def _load_matrix(fp: Path | str):
    fp = Path(fp)
    return sio.loadmat(str(fp))[fp.stem]

def _load_image(fp: Path | str):
    fp = Path(fp)
    return np.array(Image.open(fp))

def _load_annotations(fp: Path | str):
    fp = Path(fp)
    if fp.suffix == '.mat':
        return _load_matrix(Path(fp))
    else:
        return _load_image(Path(fp))

def _load_feature(fp: Path | str):
    fp = Path(fp)
    matrix = _load_matrix(fp)
    name = fp.stem

    if matrix.ndim == 2:
        matrix = np.expand_dims(matrix, axis=-1)

    return Feature(name, matrix)

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


class Feature:
    def __init__(self, name, data: np.ndarray, parent_feature=None, metadata=None):
        self.name = name
        self.data = data
        self.parent_feature = parent_feature
        self.metadata = {} if metadata is None else metadata

    @property
    def size(self):
        return self.data.shape[-1]

    def map(self, fn, name):
        data = fn(self.data)
        return Feature(name, data, self)


class Dataset:
    def __init__(self, name: str, features: typing.Sequence[Feature], labels_train, labels_test=None, config: DatasetConfig | None=None):
        self.name = name
        self.features = {feature.name: feature for feature in features}
        self.labels_train = labels_train
        self.labels_test = labels_test
        self.image_dimensions = features[0].data.shape[:2] # todo: make sure this is the same for all features
        self.config = config

    def data(self, features: list[str] | None = None, missing_value=0):
        h, w = self.image_dimensions

        if features is None:
            features = self.features.keys()

        feature_map = []
        current_channel = 0
        for feature in features:
            feat = self.feature(feature)
            if feat is None:
                print(
                    f'Feature "{feature}" not in dataset. Adding missing values ({missing_value} at channel {current_channel}).')
                feature_map.append(np.full((h, w, 1), missing_value))
                current_channel += 1
            else:
                feature_map.append(feat.data)
                current_channel += feat.size

        return np.concatenate(feature_map, axis=-1)

    def feature(self, name):
        return self.features.get(name)

    @classmethod
    def from_config(cls, config: DatasetConfig):
        base_path = Path(config.base_dir)

        features = [_load_feature(base_path / file_name) for file_name in config.feature_files]
        labels_train = _load_annotations(base_path / config.labels_file)
        labels_test = _load_annotations(base_path / config.labels_file_test) if config.labels_file_test else None

        return cls(
            name=config.name,
            features=features,
            labels_train=labels_train,
            labels_test=labels_test,
            config=config,
        )

    @classmethod
    def from_json(cls, json_path: str):
        return cls.from_config(DatasetConfig.from_json(json_path))


def reduce_dimensions(data: np.ndarray, num_components=15, valid_indices=None, return_pca=False):
    X = np.array(data)
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


def preprocess(dataset: Dataset):
    labels_train = dataset.labels_train
    labels_test = dataset.labels_test

    hs_feat = dataset.feature("data_HS_LR")
    hs = hs_feat.data
    n_components = 15
    train_indices = labels_train != 0

    hs_pca, pca = reduce_dimensions(hs, n_components, train_indices, return_pca=True)
    name = 'data_HS_LR_pca15'
    dataset.features[name] = Feature(name, hs_pca, parent_feature=hs_feat)

    features = [
        'data_DSM',
        'data_HS_LR_pca15',
        'data_SAR_HR',
    ]
    image = dataset.data(features)

    X_train, y_train = patchify(image, labels_train)
    X_test, y_test = patchify(image, labels_test)
    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    json_path = "data/.berlin.json"
    cfg = DatasetConfig.from_json(json_path)
    dataset = Dataset.from_json(json_path)
    X_train, X_test, y_train, y_test = preprocess(dataset)

    json_path = "data/.houston.json"
    cfg = DatasetConfig.from_json(json_path)
    dataset = Dataset.from_json(json_path)
    dataset.labels_train
    X_train, X_test, y_train, y_test = preprocess(dataset)
