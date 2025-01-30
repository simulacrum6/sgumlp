import functools
import typing
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.io as sio
import sklearn
from PIL import Image
from numpy.lib.stride_tricks import sliding_window_view
from sklearn.decomposition import PCA
import rasterio
from tqdm import tqdm

import torch

from .config import DatasetConfig


def _load_matrix(fp: Path | str):
    fp = Path(fp)
    return sio.loadmat(str(fp))[fp.stem]


def _load_image(fp: Path | str):
    fp = Path(fp)
    return np.array(Image.open(fp))


def _load_rasterio(fp: Path | str):
    with open(fp, "rb") as f:
        with rasterio.open(f) as src:
            return src.read()


def _load_source(fp: Path | str):
    fp = Path(fp)
    if fp.suffix == ".mat":
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


class PatchDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        data_root,
        input_img_fps,
        target_img_fps,
        patch_size=9,
        pad_value=0,
        mmap_dir=None,
        mmap_max_images=None,
        mmap_init_flush_every=100,
        mmap_init_overwrite=False,
    ):
        super().__init__()
        self.data_root = Path(data_root)
        self.input_img_fps = np.asarray(input_img_fps)
        self.target_img_fps = np.asarray(target_img_fps)

        self.patch_size = patch_size
        self.pad_value = pad_value

        self._sample_img, self._sample_target = self._load_images(input_img_fps[0], target_img_fps[0])
        if not (self._sample_img.shape[1:] == self._sample_target.shape[1:]):
            raise ValueError("Shape of input and target images do not match")

        if mmap_dir is None:
            mmap_dir = self.data_root / "mmap"

        if mmap_max_images is None:
            mmap_max_images = self.num_images

        self.mmap_max_images = mmap_max_images
        self.mmap_init_flush_every = mmap_init_flush_every
        self.mmap_init_overwrite = mmap_init_overwrite
        if self.mmap_max_images > 0:
            self.mmap_dir = Path(mmap_dir)
            self._mmap_images, self._mmap_targets, self._mmap_images_path, self._mmap_targets.path = self._init_mmaps(flush_every=mmap_init_flush_every, overwrite=mmap_init_overwrite)
        else:
            self._mmap_images, self._mmap_targets, self._mmap_images_path, self._mmap_targets.path, self.mmap_dir = None

    @property
    def pad_size(self):
        return self.patch_size // 2

    @property
    def num_pixels(self):
        return self.height * self.width

    @property
    def num_images(self):
        return self.input_img_fps.shape[0]

    @property
    def height(self):
        return self._sample_img.shape[1] - 2 * self.pad_size

    @property
    def width(self):
        return self._sample_img.shape[2] - 2 * self.pad_size

    @property
    def image_channels(self):
        return self._sample_img.shape[0]

    @property
    def target_channels(self):
        return self._sample_target.shape[0]

    @property
    def image_dimensions(self):
        return self.image_channels, self.height, self.width

    @property
    def target_dimensions(self):
        return self.target_channels, self.height, self.width

    def __len__(self):
        return self.num_images * self.num_pixels

    def _init_mmaps(self, flush_every=100, overwrite=False):
        map_params = [
            ("input.mmap", (self.mmap_max_images, *self._sample_img.shape), self._sample_img.dtype),
            ("target.mmap", (self.mmap_max_images, *self._sample_target.shape), self._sample_target.dtype),
        ]
        self.mmap_dir.mkdir(parents=True, exist_ok=True)

        maps = []
        paths = []
        for filename, shape, dtype in map_params:
            filepath = self.mmap_dir / filename
            paths.append(filepath)
            if not filepath.exists():
                arr = np.memmap(
                    filepath,
                    dtype=dtype,
                    mode='w+',
                    shape=shape
                )
                arr.flush()

            else:
                arr = np.memmap(
                    filepath,
                    dtype=dtype,
                    mode='r+',
                    shape=shape
                )
            maps.append(arr)

        mmap_images, mmap_targets = maps

        if not all([path.exists() for path in paths]) or overwrite:
            for i in tqdm(range(self.mmap_max_images), total=self.mmap_max_images, unit="image pairs"):
                img, target = self.load_images(i)
                mmap_images[i] = img
                mmap_targets[i] = target

                if i % flush_every == 0:
                    mmap_images.flush()
                    mmap_targets.flush()

            mmap_images.flush()
            mmap_targets.flush()

        path_mmap_img, path_mmap_target = paths
        return mmap_images, mmap_targets, path_mmap_img, path_mmap_target

    def _get_paths(self, i):
        return (self.input_img_fps[i], self.target_img_fps[i])

    def _load_images(self, image_fp: str, target_img_fp: str):
        image_fp = self.data_root / image_fp
        target_img_fp = self.data_root / target_img_fp
        img = _load_rasterio(image_fp)
        target = _load_rasterio(target_img_fp)

        pad_args = dict(
            pad_width=(
                (0, 0),
                (self.pad_size, self.pad_size),
                (self.pad_size, self.pad_size),
            ),
            mode="constant",
            constant_values=self.pad_value,
        )
        return np.pad(img, **pad_args), np.pad(target, **pad_args)

    def load_images(self, i):
        image_fp, target_img_fp = self._get_paths(i)
        return self._load_images(image_fp, target_img_fp)

    def __getitem__(self, idx):
        i = idx // self.num_pixels
        patch_idx = idx % self.num_pixels
        h = patch_idx // self.height
        h_ = h + self.patch_size
        w = patch_idx % self.width
        w_ = w + self.patch_size

        if i < self.mmap_max_images:
            img, target = self._mmap_images[i], self._mmap_targets[i]
        else:
            img, target = self.load_images(i)
        patch = img[:, h:h_, w:w_]
        mask = target[:, h, w]

        return torch.tensor(patch).float(), torch.tensor(mask).float()


def _is_valid(a, na_value):
    return a != na_value


def patchify(
    image: np.ndarray,
    labels=None,
    patch_size: int = 9,
    pad_value=0,
    na_value=0,
    only_valid=True,
):
    """Extract square patches centered on each pixel of a 2D or N-D image array.

    Args:
        image: Input array of shape (H, W, ...) where H and W are spatial dimensions.
            Additional dimensions after the first two are preserved as features.
        labels: Label array of shape (H, W). When provided with only_valid=True, patches
            will only be extracted for pixels that have labels (where labels != na_value).
            Defaults to None.
        patch_size: Width and height of patches. Must be odd to ensure patches are centered.
            Defaults to 9.
        pad_value: Value used to pad borders of image before extracting patches.
            Defaults to 0.
        na_value: Value in labels array that indicates pixels without a valid label.
            Defaults to 0.
        only_valid: If True and labels is provided, only extract patches centered on labeled
            pixels (where labels != na_value). If False or labels=None, extract patches
            for all pixels. Defaults to True.

    Returns:
        A tuple of (patches, labels):
            patches: Extracted patches with shape:
                - (N, patch_size, patch_size, ...) if only_valid=True and labels provided,
                  where N is number of labeled pixels
                - (H*W, patch_size, patch_size, ...) otherwise.
            labels: If labels provided, returns corresponding label for each patch:
                - (N,) if only_valid=True and labels provided,
                  where N is number of labeled pixels
                - (H*W,) otherwise.
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
    if image.ndim < 2:
        raise ValueError("image must have at least 2 dimensions")

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
    return image


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
    def __init__(
        self,
        name: str,
        features: typing.Sequence[Feature],
        labels_train,
        labels_test=None,
        config: DatasetConfig | None = None,
        na_value=0,
    ):
        self.name = name
        self.features = {feature.name: feature for feature in features}
        self.labels_train = labels_train
        self.labels_test = labels_test
        self.image_dimensions = features[0].data.shape[:2]
        self.na_value = na_value
        for feature in features:
            if feature.data.shape[:2] != self.image_dimensions:
                raise ValueError(
                    "All features must have same image dimensions (0 and 1)"
                )
        self.config = config

        self._label_encoder = sklearn.preprocessing.LabelEncoder()
        mask = _is_valid(self.labels_train, self.na_value)
        self._label_encoder.fit(self.labels_train[mask].ravel())
        self.num_labels = len(self._label_encoder.classes_)

    def data(self, features: list[str] | None = None, split=None):
        h, w = self.image_dimensions

        if features is None:
            features = self.features.keys()

        feature_map = []
        current_channel = 0
        for feature in features:
            feat = self.feature(feature)
            if feat is None:
                print(
                    f'Feature "{feature}" not in dataset. Adding na_values ({self.na_value} at channel {current_channel}).'
                )
                feature_map.append(np.full((h, w, 1), self.na_value))
                current_channel += 1
            else:
                feature_map.append(feat.data)
                current_channel += feat.size

        feature_map = np.concatenate(feature_map, axis=-1)

        if split is not None:
            mask = self.split_mask(split)
            return feature_map[mask]

        return feature_map

    def feature(self, name):
        return self.features.get(name)

    def labels(self, split="train"):
        return self.labels_train if split == "train" else self.labels_test

    def split_mask(self, split="train"):
        labels = self.labels(split)
        return labels != self.na_value

    def targets(self, split="train"):
        labels = self.labels(split)
        mask = self.split_mask(split)
        return self._label_encoder.transform(labels[mask])

    def label_to_id(self, labels):
        return self._label_encoder.transform(labels)

    def id_to_label(self, ids):
        return self._label_encoder.inverse_transform(ids)

    @classmethod
    def from_config(cls, config: DatasetConfig):
        base_path = Path(config.base_dir)

        features = [
            _load_feature(base_path / file_name) for file_name in config.feature_files
        ]
        labels_train = _load_source(base_path / config.labels_file)
        labels_test = (
            _load_source(base_path / config.labels_file_test)
            if config.labels_file_test
            else None
        )

        return cls(
            name=config.name,
            features=features,
            labels_train=labels_train,
            labels_test=labels_test,
            config=config,
        )

    @classmethod
    def from_json(cls, json_path: str | Path):
        return cls.from_config(DatasetConfig.from_json(json_path))


def reduce_dimensions(
    image: np.ndarray, num_components=15, valid_indices=None, return_pca=False
):
    X = np.array(image, copy=True)
    h, w, c = X.shape

    if valid_indices is not None:
        X = X[valid_indices]

    if X.ndim > 2:
        X = X.reshape(-1, c)

    pca = PCA(n_components=num_components, whiten=True).fit(X)
    X_reduced = pca.transform(image.reshape(-1, c)).reshape(h, w, num_components)

    if return_pca:
        return X_reduced, pca

    return X_reduced


def preprocess(
    dataset: Dataset,
    image_dtype=np.float32,
    num_hs_components=15,
    return_train_test=True,
):
    train_mask = dataset.split_mask("train")
    test_mask = dataset.split_mask("test")

    if num_hs_components is not None:
        hs_feat = dataset.feature("data_HS_LR")
        hs = hs_feat.data
        n_components = 15
        hs_pca, pca = reduce_dimensions(hs, n_components, train_mask, return_pca=True)
        name = "data_HS_LR_pca15"
        dataset.features[name] = Feature(name, hs_pca, parent_feature=hs_feat)

        image = dataset.data(
            [
                "data_DSM",
                "data_HS_LR_pca15",
                "data_SAR_HR",
            ]
        )
    else:
        image = dataset.data()

    X = patchify(image)

    if not return_train_test:
        X = np.concatenate([X[train_mask], X[test_mask]]).astype(image_dtype)
        y = np.concatenate([dataset.targets("train"), dataset.targets("test")])
        return X, y

    return (
        X[train_mask].astype(image_dtype),
        X[test_mask].astype(image_dtype),
        dataset.targets("train"),
        dataset.targets("test"),
    )
