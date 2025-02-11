import torch

activations = {
    "gelu": torch.nn.GELU,
    "relu": torch.nn.ReLU,
}


def DepthWiseConv2d(in_channels, kernel_size, num_kernels=1, *args, **kwargs):
    """
    Creates a depthwise 2D convolutional layer.

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
    in parallel and adds their outputs.

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
    Implements a Spatial Gated Unit.

    This unit splits input features into two paths and modulates outputs of one path by learned interactions across
    features of the second.

    Args:
        in_features (int): Number of input features (must be even as it's split in half)
        sequence_length (int): Length of input sequence for weight matrix dimensions
        epsilon (float, optional): Range for uniform weight initialization. Defaults to 1e-3
        channel_dim (int, optional): Dimension along which to split channels. Defaults to -1

    Forward Args:
        x (torch.Tensor): Input tensor to be gated [batch_size, sequence_length, in_features]

    Returns:
        torch.Tensor: Gated output tensor with shape [batch_size, sequence_length, in_features // 2]

    See Also:
        - "Pay Attention to MLPs" by Liu et al. 2021, https://arxiv.org/abs/2105.08050
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
    A Multi-Layer Perceptron block that optionally incorporates a Spatial Gated Unit between linear layers.

    Args:
        in_features (int): Number of input features
        hidden_features (int): Number of hidden features
        sequence_length (int, optional): Required when use_sgu=True for SGU dimensions
        activation (str, optional): Name of activation function ('gelu' or 'relu'). Defaults to 'gelu'
        use_sgu (bool, optional): Whether to use Spatial Gated Unit. Defaults to False
        out_features (int, optional): Number of output features. Defaults to in_features if None
        dropout (float, optional): Dropout probability. Defaults to 0.0

    Forward Args:
        x (torch.Tensor): Input tensor [batch_size, sequence_length, in_features]

    Returns:
        torch.Tensor: Transformed output with shape [batch_size, sequence_length, out_features]

    Raises:
        ValueError: If sequence_length is None when use_sgu=True
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
    MLP-Mixer block combining token and channel mixing.
    It can optionally use Spatial Gated Units in MixerBlocks.

    Args:
        in_features (int): Number of input features per token
        sequence_length (int): Number of tokens in the sequence
        hidden_features_channel (int): Hidden dimension size for channel mixing
        hidden_features_sequence (int): Hidden dimension size for token mixing
        activation (str, optional): Activation function name. Defaults to 'gelu'
        use_sgu (bool, optional): Whether to use SGU in mixing operations. Defaults to False
        dropout (float, optional): Dropout probability. Defaults to 0.0

    Forward Args:
        x (torch.Tensor): Input tensor [batch_size, sequence_length, in_features]

    Returns:
        torch.Tensor: Processed tensor with same shape as input after both mixing operations
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
    Calculates the number of tokens when dividing an image into square patches.

    Args:
        height (int): Image height in pixels
        width (int): Image width in pixels
        patch_size (int): Size of each square patch

    Returns:
        int: Total number of tokens (patches)

    Note:
        Assumes the image dimensions are perfectly divisible by patch_size
    """
    return height * width // patch_size**2


class MLPMixer(torch.nn.Module):
    """
    A MLP-based vision architecture that processes images using token and channel mixing.

    This implementation splits images into patches, projects them to a token embedding space,
    and processes them through multiple MLPMixer blocks. It can optionally use Spatial Gated
    Units instead of standard MLPs.

    Args:
        image_dimensions (tuple): Input image shape as (height, width, channels)
        patch_size (int): Size of square patches to divide the image
        token_features (int): Dimension of token embedding space
        mixer_features_channel (int): Hidden dimension for channel-mixing MLPs
        mixer_features_sequence (int): Hidden dimension for token-mixing MLPs
        mixer_use_sgu (bool, optional): Whether to use SGU in mixer blocks. Defaults to False
        num_blocks (int, optional): Number of sequential mixer blocks. Defaults to 1
        activation (str, optional): Activation function type. Defaults to 'gelu'
        dropout (float, optional): Dropout probability. Defaults to 0.0

    Forward Args:
        x (torch.Tensor): Input image [batch_size, channels, height, width]

    Returns:
        torch.Tensor: Processed features [batch_size, n_tokens, token_features]

    See Also:
        - "MLP-Mixer: An all-MLP Architecture for Vision" by Tolstikhin et al. 2021, https://arxiv.org/abs/2105.01601
        - "Pay Attention to MLPs" by Liu et al. 2021, https://arxiv.org/abs/2105.08050
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
    """
    Classification head that pools tokens and applies linear classification.

    Args:
        num_classes (int): Number of output classes
        in_features (int): Number of input features per token
        dropout (float, optional): Dropout probability. Defaults to 0.0

    Forward Args:
        x (torch.Tensor): Input features [batch_size, n_tokens, in_features]

    Returns:
        torch.Tensor: Classification logits [batch_size, num_classes]
    """
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
    MLP-Mixer variant using Spatial Gated Units (SGU) during mixing and parallel depthwise convolutions with residual
    connections for preprocessing.

    This implementation allows setting the weight of the residual connection in preprocessing and optionally allows
    to make it learnable.

    Args:
        input_dimensions (tuple): Input patch dimensions as (height, width, channels)
        token_features (int): Dimension of the token embedding space
        mixer_features_channel (int): Hidden dimension for channel-mixing SGU/MLP blocks
        mixer_features_sequence (int): Hidden dimension for token-mixing SGU/MLP blocks
        mixer_use_sgu (bool, optional): Whether to use SGU blocks instead of standard MLPs. Defaults to True
        dwc_kernels (tuple, optional): Kernel sizes for parallel depthwise convolutions. Defaults to (1, 3, 5)
        num_blocks (int, optional): Number of sequential mixer blocks. Defaults to 1
        activation (str, optional): Activation function for blocks. Defaults to 'gelu'
        residual_weight (float, optional): Scaling factor for residual connection. Defaults to 2
        learnable_residual (bool, optional): Whether residual weight is learnable. Defaults to False
        embedding_kernel_size (int, optional): Kernel size for token embedding conv layer. Defaults to 1
        num_classes (int, optional): Number of output classes. If 0, returns embeddings. Defaults to 0
        dropout (float, optional): Dropout rate for mixer blocks and classifier. Defaults to 0.0
        channels_first (bool, optional): Whether input has channels in first dimension. Defaults to False

    Forward Args:
        x (torch.Tensor): Input tensor of shape:
            (batch_size, channels, height, width) if channels_first=True
            (batch_size, height, width, channels) if channels_first=False

    Returns:
        torch.Tensor: Output tensor of shape:
            (batch_size, num_classes) if num_classes > 0
            (batch_size, n_tokens, token_features) if num_classes = 0
            where n_tokens = (height - embedding_kernel_size + 1)^2

    Raises:
        ValueError: If patch dimensions are not square
        ValueError: If patch height/width is not odd
        ValueError: If embedding kernel size is larger than patch size

    See Also:
        - "Spatial Gated Multi-Layer Perceptron for Land Use and Land Cover Mapping" by Jamali et al. 2024, https://github.com/aj1365/SGUMLP/blob/main/Spatial_Gated_Multi-Layer_Perceptron_for_Land_Use_and_Land_Cover_Mapping.pdf
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
        channels_first=False,
    ):
        super().__init__()
        self.input_dimensions = input_dimensions

        # validate dimensions
        patch_height, patch_width, patch_channels = input_dimensions
        if patch_height != patch_width:
            raise ValueError("Patch size must be square")
        if patch_height % 2 == 0:
            raise ValueError("Patch height must be odd")

        self.channels_first = channels_first
        self.num_classes = num_classes
        self.patch_size = patch_height
        self.embedding_kernel_size = embedding_kernel_size
        self.embedding_patch_size = patch_height - embedding_kernel_size + 1
        self.n_tokens = self.embedding_patch_size * self.embedding_patch_size
        self.n_channels = token_features
        self.dropout_rate = dropout

        self.output_dimensions = (self.n_tokens, self.n_channels) if num_classes == 0 else (num_classes,)

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
        if not self.channels_first:
            x = x.moveaxis(-1, 1)
        residual = x
        x = self.dwc(x)
        x = x + (residual * self.residual_weight)
        x = self.token_embedding(x)
        x = x.reshape(b, self.n_channels, self.n_tokens).transpose(1, 2)
        x = self.mixer_blocks(x)
        x = self.head(x)
        return x



