import torch
import lightning
import torchmetrics

activations = {
    "gelu": torch.nn.GELU,
    "relu": torch.nn.ReLU,
}


def DepthWiseConv2d(in_channels, kernel_size, num_kernels=1, *args, **kwargs):
    """
    Creates a depthwise 2D convolutional layer with optional channel multiplication.

    Args:
       in_channels (int): Number of input channels
       kernel_size (int or tuple): Size of the convolving kernel
       num_kernels (int, optional): Number of kernels per input channel. Defaults to 1
       *args: Additional positional arguments passed to Conv2d
       **kwargs: Additional keyword arguments passed to Conv2d

    Returns:
       torch.nn.Conv2d: A depthwise convolutional layer where each input channel is
       convolved with its own set of filters
    """
    return torch.nn.Conv2d(
        in_channels,
        in_channels * num_kernels,
        kernel_size=kernel_size,
        groups=in_channels,
        padding="same",
        *args,
        **kwargs,
    )


class ParallelDepthwiseConv2d(torch.nn.Module):
    """
    Implements a multi-kernel depthwise convolution module that applies multiple kernel sizes
    in parallel and combines their outputs.

    Args:
        in_channels (int): Number of input channels
        kernel_sizes (list or tuple): List of kernel sizes to use for parallel convolutions
        *args: Additional positional arguments passed to DepthWiseConv2d
        **kwargs: Additional keyword arguments passed to DepthWiseConv2d

    Forward Args:
        x (torch.Tensor): Input tensor of shape [batch_size, channels, height, width]

    Returns:
        torch.Tensor: Output tensor after applying all kernels and summing results
    """

    def __init__(self, in_channels, kernel_sizes, *args, **kwargs):
        super().__init__()
        self.kernels = torch.nn.ModuleList()
        for kernel_size in kernel_sizes:
            self.kernels.append(
                DepthWiseConv2d(in_channels, kernel_size, *args, **kwargs)
            )

    def forward(self, x):
        return torch.stack([kernel(x) for kernel in self.kernels], dim=1).sum(dim=1)


class SpatialGatedUnit(torch.nn.Module):
    """
    Implements a Spatial Gated Unit that splits input features into two paths,
    applying spatial gating to control information flow.

    Args:
        in_features (int): Number of input features (must be even)
        sequence_length (int): Length of input sequence
        epsilon (float, optional): Range for weight initialization (-epsilon, epsilon). Defaults to 1e-3
        channel_dim (int, optional): Dimension along which to split channels. Defaults to -1

    Forward Args:
        x (torch.Tensor): Input tensor to be gated

    Returns:
        torch.Tensor: Gated output tensor with half the channels of the input
    """

    def __init__(self, in_features, sequence_length, epsilon=1e-3, channel_dim=-1):
        super().__init__()
        self.channel_dim = channel_dim
        self.epsilon = epsilon

        self.layer_norm = torch.nn.LayerNorm(in_features // 2)
        self.weights = torch.nn.Parameter(torch.empty(sequence_length, sequence_length))
        self.bias = torch.nn.Parameter(torch.ones(sequence_length, 1))

        torch.nn.init.uniform_(self.weights, -self.epsilon, self.epsilon)

    def forward(self, x):
        u, v = torch.chunk(x, 2, dim=self.channel_dim)
        v = self.layer_norm(v)
        return u * ((self.weights @ v) + self.bias)


class MLPBlock(torch.nn.Module):
    """
    A Multi-Layer Perceptron block that incorporates a Spatial Gated Unit between linear layers.

    Args:
        in_features (int): Number of input features
        hidden_features (int): Number of hidden features before the SGU
        sequence_length (int): Length of input sequence for SGU
        activation (str, optional): Activation function to use. Defaults to 'gelu'
        out_features (int, optional): Number of output features. Defaults to in_features if None

    Forward Args:
        x (torch.Tensor): Input tensor

    Returns:
        torch.Tensor: Transformed output tensor after spatial gating
    """

    def __init__(
        self,
        in_features,
        hidden_features,
        sequence_length=None,
        activation="gelu",
        use_sgu=False,
        out_features=None,
        dropout=0.0,
    ):
        super().__init__()
        if sequence_length is None and use_sgu:
            raise ValueError("When using SGU, sequence_length must be specified")

        if out_features is None:
            out_features = in_features

        self.activation = activations.get(activation, torch.nn.GELU)()
        self.dropout = torch.nn.Dropout(dropout)
        self.mlp1 = torch.nn.Linear(in_features, hidden_features)

        if use_sgu:
            self.sgu = SpatialGatedUnit(hidden_features, sequence_length)
            self.mlp2 = torch.nn.Linear(hidden_features // 2, out_features)
        else:
            self.sgu = torch.nn.Identity()
            self.mlp2 = torch.nn.Linear(hidden_features, out_features)

    def forward(self, x):
        x = self.mlp1(x)
        x = self.sgu(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.mlp2(x)
        x = self.dropout(x)
        return x


class MLPMixerBlock(torch.nn.Module):
    """
    A block in the MLP-Mixer architecture that performs both token and channel mixing operations.

    This block applies two types of mixing:
    1. Token mixing: Processes relationships between different tokens at the same feature
    2. Channel mixing: Processes relationships between different features for each token

    Args:
        in_features (int): Number of input features per token
        sequence_length (int): Number of tokens in the sequence
        hidden_features_channel (int): Number of hidden dimensions in the channel mixing MLP
        hidden_features_sequence (int): Number of hidden dimensions in the token mixing MLP
        activation (str, optional): Name of activation function to use. Defaults to 'gelu'
        mlp_block (str, optional): Type of MLP block implementation ('mlp' or 'sgu'). Defaults to 'mlp'

    Input Shape:
        - Input: (batch_size, sequence_length, in_features)
        - Output: (batch_size, sequence_length, in_features)
    """

    def __init__(
        self,
        in_features,
        sequence_length,
        hidden_features_channel,
        hidden_features_sequence,
        activation="gelu",
        use_sgu=False,
        dropout=0.0,
    ):
        super().__init__()

        self.dropout = torch.nn.Dropout(dropout)
        self.layer_norm = torch.nn.LayerNorm([in_features])
        self.channel_mixer = MLPBlock(
            in_features=in_features,
            hidden_features=hidden_features_channel,
            sequence_length=sequence_length,
            activation=activation,
            dropout=dropout,
            use_sgu=use_sgu,
        )
        self.token_mixer = MLPBlock(
            in_features=sequence_length,
            hidden_features=hidden_features_sequence,
            sequence_length=in_features,
            activation=activation,
            dropout=dropout,
            use_sgu=use_sgu,
        )

    def forward(self, x):
        residual = x

        x = self.layer_norm(x).transpose(1, 2)
        x = self.token_mixer(x).transpose(1, 2)
        x = self.dropout(x)

        residual = residual + x

        x = self.layer_norm(residual)
        x = self.channel_mixer(x)
        x = self.dropout(x)
        return residual + x


def calc_n_tokens(height, width, patch_size):
    """
    Calculates the number of tokens after splitting an image into patches.

    Args:
        height (int): Height of the input image
        width (int): Width of the input image
        patch_size (int): Size of each square patch

    Returns:
        int: Number of tokens (patches) the image will be divided into
    """
    return height * width // patch_size**2


class MLPMixer(torch.nn.Module):
    """
    MLP-Mixer architecture for processing images.

    The model first splits input images into patches, projects them to a token embedding space,
    and then processes them through multiple MLPMixer blocks. It relies purely on MLPs for
    both spatial (token) and channel mixing, avoiding attention mechanisms.

    Args:
        image_dimensions (tuple): Image input shape as (height, width, channels)
        patch_size (int): Size of square patches to divide the image into
        token_features (int): Number of features in token embedding space
        mixer_features_channel (int): Hidden dimension size for channel-mixing MLPs
        mixer_features_sequence (int): Hidden dimension size for token-mixing MLPs
        num_blocks (int, optional): Number of sequential mixer blocks. Defaults to 1
        activation (str, optional): Activation function for MLPs. Defaults to 'gelu'

    Input Shape:
        - Input: (batch_size, height, width, channels)
        - Output: (batch_size, n_tokens, token_features)
        where n_tokens = (height * width) / (patch_size * patch_size)
    """

    def __init__(
        self,
        image_dimensions,
        patch_size: int,
        token_features: int,
        mixer_features_channel: int,
        mixer_features_sequence: int,
        mixer_use_sgu=False,
        num_blocks=1,
        activation="gelu",
        dropout=0.0,
    ):
        super().__init__()
        h, w, c = image_dimensions
        self.n_tokens = calc_n_tokens(h, w, patch_size)
        self.n_channels = token_features

        self.token_embedding = torch.nn.Conv2d(
            c, token_features, kernel_size=patch_size, stride=patch_size
        )
        self.mixer_blocks = torch.nn.ModuleList()
        for _ in range(num_blocks):
            self.mixer_blocks.append(
                MLPMixerBlock(
                    in_features=self.n_channels,
                    hidden_features_channel=mixer_features_channel,
                    hidden_features_sequence=mixer_features_sequence,
                    sequence_length=self.n_tokens,
                    activation=activation,
                    use_sgu=mixer_use_sgu,
                    dropout=dropout,
                )
            )

    def forward(self, x):
        b = x.shape[0]
        x = self.token_embedding(x.moveaxis(-1, 1))
        x = x.reshape(b, self.n_channels, self.n_tokens).transpose(1, 2)
        for block in self.mixer_blocks:
            x = block(x)
        return x


class Classifier(torch.nn.Module):
    def __init__(self, num_classes, in_features, dropout=0.0):
        super().__init__()
        self.avg_pool_tokens = torch.nn.AdaptiveAvgPool1d(1)
        self.dropout = torch.nn.Dropout(dropout)
        self.clf = torch.nn.Linear(in_features, num_classes)

    def forward(self, x):
        x = self.avg_pool_tokens(x.transpose(-1, -2)).transpose(-1, -2).squeeze()
        x = self.dropout(x)
        return self.clf(x)


class SGUMLPMixer(torch.nn.Module):
    """
    MLP-Mixer variant using Spatial Gated Units (SGU) and parallel depthwise convolutions with residual connections.

    This implementation enhances the base MLP-Mixer through several architectural modifications:
    1. Replacing standard MLPs with Spatial Gated Units (SGU)
    2. Using parallel depthwise convolutions with multiple kernel sizes for multi-scale feature extraction
    3. Employing direct patch processing instead of image-level patching
    4. Adding a scaled residual connection (2x) around the depthwise convolution layer

    Args:
        input_dimensions (tuple): Input patch dimensions as (height, width, channels)
        token_features (int): Dimension of the token embedding space
        mixer_features_channel (int): Hidden dimension for channel-mixing SGU blocks
        mixer_features_sequence (int): Hidden dimension for token-mixing SGU blocks
        dwc_kernels (tuple, optional): Kernel sizes for parallel depthwise convolutions. Defaults to (1, 3, 5)
        num_blocks (int, optional): Number of sequential mixer blocks. Defaults to 1
        activation (str, optional): Activation function for SGU blocks. Defaults to 'gelu'

    Input / Output Shapes:
        - Input: (batch_size, patch_height, patch_width, patch_channels)
        - Output: (batch_size, n_tokens, token_features)
        where n_tokens = patch_height * patch_width

    Notes:
        - Patch dimensions must be square (height == width)
        - Patch height/width must be odd-numbered
        - All mixer blocks use SGU instead of standard MLPs for both token and channel mixing
    """

    def __init__(
        self,
        input_dimensions,
        token_features: int,
        mixer_features_channel: int,
        mixer_features_sequence: int,
        mixer_use_sgu=True,
        dwc_kernels=(1, 3, 5),
        num_blocks=1,
        activation="gelu",
        residual_weight=2,
        learnable_residual=False,
        embedding_kernel_size=1,
        num_classes=0,
        dropout=0.0,
    ):
        super().__init__()
        self.patch_dimensions = input_dimensions

        # validate dimensions
        patch_height, patch_width, patch_channels = input_dimensions
        if patch_height != patch_width:
            raise ValueError("Patch size must be square")
        if patch_height % 2 == 0:
            raise ValueError("Patch height must be odd")

        self.num_classes = num_classes
        self.patch_size = patch_height
        self.embedding_kernel_size = embedding_kernel_size
        self.embedding_patch_size = patch_height - embedding_kernel_size + 1
        self.n_tokens = self.embedding_patch_size * self.embedding_patch_size
        self.n_channels = token_features
        self.dropout_rate = dropout

        self.dwc = ParallelDepthwiseConv2d(patch_channels, dwc_kernels)
        residual_module = torch.nn.Parameter if learnable_residual else torch.nn.Buffer
        self.residual_weight = residual_module(torch.tensor(residual_weight))
        self.token_embedding = torch.nn.Conv2d(
            patch_channels,
            self.n_channels,
            kernel_size=embedding_kernel_size,
            padding="valid",
        )
        self.mixer_blocks = torch.nn.Sequential(
            *[
                MLPMixerBlock(
                    in_features=self.n_channels,
                    sequence_length=self.n_tokens,
                    hidden_features_channel=mixer_features_channel,
                    hidden_features_sequence=mixer_features_sequence,
                    activation=activation,
                    use_sgu=mixer_use_sgu,
                    dropout=self.dropout_rate,
                )
                for _ in range(num_blocks)
            ]
        )
        self.head = (
            Classifier(num_classes, self.n_channels, self.dropout_rate)
            if num_classes > 0
            else torch.nn.Identity()
        )

    def forward(self, x):
        b = x.shape[0]
        x = x.moveaxis(-1, 1)
        residual = x
        x = self.dwc(x)
        x = x + (residual * self.residual_weight)
        x = self.token_embedding(x)
        x = x.reshape(b, self.n_channels, self.n_tokens).transpose(1, 2)
        x = self.mixer_blocks(x)
        x = self.head(x)
        return x


class LitSGUMLPMixer(lightning.LightningModule):
    def __init__(self, model_params, optimizer_params, metrics=None, *args, **kwargs):
        super().__init__()
        self.save_hyperparameters(ignore=["metrics"])

        if metrics is None:
            metrics = self._default_metrics

        self.model = SGUMLPMixer(**model_params)
        self.criterion = torch.nn.CrossEntropyLoss()
        self.optimizer_cls = torch.optim.AdamW

        self.train_metrics = torchmetrics.MetricCollection(
            dict(metrics.get("train", self._default_metrics["train"]))
        )
        self.test_metrics = torchmetrics.MetricCollection(
            dict(metrics.get("test", self._default_metrics["test"]))
        )

    @property
    def _default_metrics(self):
        task = "binary" if self.model.num_classes == 2 else "multiclass"
        return dict(
            train=dict(
                accuracy=torchmetrics.Accuracy(
                    task=task, num_classes=self.model.num_classes
                ),
            ),
            test=dict(
                accuracy=torchmetrics.Accuracy(
                    task=task, num_classes=self.model.num_classes
                ),
            ),
        )

    def forward(self, x):
        return self.model(x)

    def _step(self, batch, batch_idx, split, on_step=False, on_epoch=True):
        x, y = batch
        y_hat = self.forward(x)
        loss = self.criterion(y_hat, y)

        self.log(f"{split}_loss", loss)
        metrics = self.train_metrics if split == "train" else self.test_metrics
        for name, metric in metrics.items():
            self.log(
                f"{split}_{name}",
                metric(y_hat, y),
                on_step=on_step,
                on_epoch=on_epoch,
                prog_bar=True,
            )
        return loss

    def training_step(self, batch, batch_idx):
        return self._step(batch, batch_idx, "train", on_step=True, on_epoch=False)

    def validation_step(self, batch, batch_idx):
        return self._step(batch, batch_idx, "val", on_step=False, on_epoch=True)

    def test_step(self, batch, batch_idx):
        return self._step(batch, batch_idx, "test", on_step=False, on_epoch=True)

    def configure_optimizers(self):
        return self.optimizer_cls(
            self.model.parameters(), **self.hparams.optimizer_params
        )
