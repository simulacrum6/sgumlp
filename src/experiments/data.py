from functools import partial
from pathlib import Path

import gdown
import numpy as np
import patoolib
import rasterio
import scipy.io as sio
import torch
from PIL import Image
from numpy.lib.stride_tricks import sliding_window_view
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm


def load_matrix(fp: Path | str):
    fp = Path(fp)
    return sio.loadmat(str(fp))[fp.stem]


def load_pil(fp: Path | str):
    fp = Path(fp)
    return np.array(Image.open(fp))


def load_rasterio(fp: Path | str):
    with open(fp, "rb") as f:
        with rasterio.open(f) as src:
            return src.read()


def load_image(fp: Path | str, channel_first=True):
    fp = Path(fp)
    match fp.suffix:
        case ".mat":
            img = load_matrix(fp)
        case ".tiff" | ".tif":
            img = load_rasterio(fp)
            if not channel_first:
                img = np.swapaxes(img, 0, -1)
        case _:
            img = load_pil(fp)
            if channel_first:
                img = np.swapaxes(img, 0, -1)
    return img


def load_benchmark_dataset(base_dir, feature_files, label_files, na_value=0, **kwargs):
    load = partial(load_image, channel_first=False)

    base_dir = Path(base_dir)
    feature_files = [Path(feature_file) for feature_file in feature_files]
    features = {
        file_name.stem: load(base_dir / file_name) for file_name in feature_files
    }

    if type(label_files) == str:
        label_files = [Path(label_files), None]

    if len(label_files) == 1:
        label_files = [label_files[0], None]

    file_train, file_test = label_files
    train_labels = load(base_dir / file_train)
    if file_test is not None:
        test_labels = load(base_dir / file_test)
    else:
        test_labels = np.full_like(train_labels, na_value)
    labels = {
        "train": train_labels,
        "test": test_labels,
    }
    return features, labels


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

        self._sample_img, self._sample_target = self._load_images(
            input_img_fps[0], target_img_fps[0]
        )
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
            (
                self._mmap_images,
                self._mmap_targets,
                self._mmap_images_path,
                self._mmap_targets.path,
            ) = self._init_mmaps(
                flush_every=mmap_init_flush_every, overwrite=mmap_init_overwrite
            )
        else:
            (
                self._mmap_images,
                self._mmap_targets,
                self._mmap_images_path,
                self._mmap_targets.path,
                self.mmap_dir,
            ) = (
                None,
                None,
                None,
                None,
                None,
            )

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
            (
                "input.mmap",
                (self.mmap_max_images, *self._sample_img.shape),
                self._sample_img.dtype,
            ),
            (
                "target.mmap",
                (self.mmap_max_images, *self._sample_target.shape),
                self._sample_target.dtype,
            ),
        ]
        self.mmap_dir.mkdir(parents=True, exist_ok=True)

        maps = []
        paths = []
        for filename, shape, dtype in map_params:
            filepath = self.mmap_dir / filename
            paths.append(filepath)
            if not filepath.exists():
                arr = np.memmap(filepath, dtype=dtype, mode="w+", shape=shape)
                arr.flush()

            else:
                arr = np.memmap(filepath, dtype=dtype, mode="r+", shape=shape)
            maps.append(arr)

        mmap_images, mmap_targets = maps

        if not all([path.exists() for path in paths]) or overwrite:
            for i in tqdm(
                range(self.mmap_max_images),
                total=self.mmap_max_images,
                unit="image pairs",
            ):
                img, target = self.load_image_and_target(i)
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
        img = load_rasterio(image_fp)
        target = load_rasterio(target_img_fp)

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

    def load_image_and_target(self, i):
        image_fp, target_img_fp = self._get_paths(i)
        return self._load_images(image_fp, target_img_fp)

    def get_image_and_target(self, i: int):
        if i < self.mmap_max_images:
            img = self._mmap_images[i].copy()
            target = self._mmap_targets[i].copy()
            return img, target
        return self.load_image_and_target(i)

    def __getitem__(self, idx):
        i = idx // self.num_pixels
        patch_idx = idx % self.num_pixels
        h = patch_idx // self.height
        h_ = h + self.patch_size
        w = patch_idx % self.width
        w_ = w + self.patch_size

        if i < self.mmap_max_images:
            patch = self._mmap_images[i, :, h:h_, w:w_].copy()
            mask = self._mmap_targets[i, :, h, w].copy()
        else:
            img, target = self.get_image_and_target(i)
            patch = img[:, h:h_, w:w_]
            mask = target[:, h, w]

        return torch.tensor(patch).float(), torch.tensor(mask).float()


def _is_valid(a, na_value):
    return a != na_value


def patchify(
    image: np.ndarray,
    patch_size: int = 9,
    pad_value=0,
):
    if patch_size % 2 == 0:
        raise ValueError("patch_size must be odd")
    offset = patch_size // 2

    image = np.array(image, copy=True)
    if image.ndim < 3:
        raise ValueError("image must have at least 3 dimensions (H, W, C) or 4 dimensions (N, H, W, C)")

    if image.ndim == 3:
        image = np.expand_dims(image, 0)

    # configure sliding windows
    pad_width = [(0, 0), (offset, offset), (offset, offset)]
    window_shape = [1, patch_size, patch_size]

    # include feature dimensions
    n_feature_dims = image.ndim - 3
    pad_width.extend([(0, 0)] * n_feature_dims)
    window_shape.extend(image.shape[3:])

    # extract patches
    image = np.pad(
        image, pad_width=pad_width, mode="constant", constant_values=pad_value
    )
    image = sliding_window_view(image, window_shape=window_shape).squeeze()
    return image


def reduce_dimensions(
    image: np.ndarray, num_components=15, channel_dim=-1, train_mask=None, pca=None
):
    channels_last = channel_dim == -1 or channel_dim == (image.ndim - 1)

    if not channels_last:
        image = np.swapaxes(image, channel_dim, -1)
        if train_mask is not None and train_mask.shape == image.shape:
            train_mask = np.swapaxes(train_mask, channel_dim, -1)

    image_dimensions = image.shape[:-1]
    c = image.shape[-1]

    if train_mask is not None:
        X = image[train_mask]
    else:
        X = image

    X = X.reshape(-1, c)

    if pca is None:
        pca = PCA(n_components=num_components, whiten=True).fit(X)

    X = pca.transform(image.reshape(-1, c)).reshape(*image_dimensions, num_components)

    if not channels_last:
        X = np.swapaxes(X, -1, channel_dim)

    return X, pca


def preprocess(
    features: dict[str, np.ndarray],
    labels: dict[str, np.ndarray],
    image_dtype=np.float32,
    features_to_process: list[str] | None = None,
    patch_size: int = 9,
    na_value: int = 0,
    num_components=15,
    channel_dim=-1,
    pcas=None,
    label_encoder=None,
):
    image_dimensions = labels["train"].shape[:2]
    num_pixels = np.prod(image_dimensions)

    valid_labels = []
    idxs = []
    for split, targets in labels.items():
        mask = np.asarray(targets != na_value)
        valid_labels.append(targets[mask])
        idxs.append(np.where(mask.reshape(num_pixels)))

    if label_encoder is None:
        label_encoder = LabelEncoder().fit(np.concatenate(valid_labels, axis=0).ravel())

    y_train, y_test = [label_encoder.transform(y) for y in valid_labels]
    idxs_train, idxs_test = idxs
    y = np.full(num_pixels, -1, dtype=int)
    y[idxs_train] = y_train
    y[idxs_test] = y_test

    if features_to_process is None:
        features_to_process = []
    features_to_process = set(features_to_process)

    if pcas is None:
        pcas = {}

    X = []
    for name, data in features.items():
        data = data.reshape(num_pixels, -1)
        if name in features_to_process:
            pca = pcas.get(name)
            data, pca = reduce_dimensions(
                data,
                num_components,
                channel_dim=channel_dim,
                train_mask=idxs_train,
                pca=pca,
            )
            pcas[name] = pca
        X.append(data)

    X = np.concatenate(X, axis=channel_dim)
    channels = X.shape[-1]
    X = X.reshape(-1, *image_dimensions, channels)
    X = patchify(X, patch_size=patch_size, pad_value=na_value).reshape(-1, patch_size, patch_size, channels)

    return (X.astype(image_dtype), y, (idxs_train, idxs_test), label_encoder, pcas)


def get_dataloader(dataset, batch_size, idxs=None, shuffle=False, num_workers=1):
    sampler = None
    if idxs is not None:
        sampler = torch.utils.data.SubsetRandomSampler(idxs)
        shuffle = None
    return torch.utils.data.DataLoader(
        dataset,
        num_workers=num_workers,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
    )


def to_torch_dataset(X, y):
    return torch.utils.data.TensorDataset(torch.from_numpy(X), torch.from_numpy(y))


def load_and_preprocess_dataset(dataset_cfg: dict, pcas=None):
    dataset = load_benchmark_dataset(
        **dataset_cfg["files"], na_value=dataset_cfg.get("na_value", 0)
    )
    X, y, (train_idx, test_idx), label_encoder, pcas = preprocess(
        *dataset, **dataset_cfg["preprocessing"], pcas=pcas
    )
    n, p, _, c = X.shape
    X_train = X[train_idx]
    y_train = y[train_idx]
    X_test = X[test_idx]
    y_test = y[test_idx]
    dataset_train = torch.utils.data.TensorDataset(
        torch.from_numpy(X_train), torch.from_numpy(y_train)
    )
    dataset_test = torch.utils.data.TensorDataset(
        torch.from_numpy(X_test), torch.from_numpy(y_test)
    )

    return {
        "name": dataset_cfg["name"],
        "train": dataset_train,
        "test": dataset_test,
        "num_labels": len(label_encoder.classes_),
        "input_dimensions": X_train.shape[1:],
        "pcas": pcas,
        "label_encoder": label_encoder,
    }


def download_benchmark_datasets(data_dir="data", overwrite=False):
    data_path = Path(data_dir)
    datasets_dir = data_path / "Datasets"
    if datasets_dir.exists() and not overwrite:
        print("Dataset already exists, skipping download. If you want to overwrite it, pass overwrite=True or delete it.")
        return

    data_path.mkdir(parents=True, exist_ok=True)
    file_id = "1dLJJrNJpQoQeDHybs37iGxmrSU6aP2xv"  # https://drive.usercontent.google.com/download?id=1dLJJrNJpQoQeDHybs37iGxmrSU6aP2xv&export=download
    file_path = data_path / (file_id + ".rar")
    try:
        gdown.download(id=file_id, output=str(file_path), quiet=False)
        patoolib.extract_archive(str(file_path), outdir=str(data_path))
    finally:
        file_path.unlink(missing_ok=True)
