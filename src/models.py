import torch


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
        **kwargs
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
    A basic Multi-Layer Perceptron block with two linear layers and an activation function.

    Args:
        in_features (int): Number of input features
        hidden_features (int): Number of hidden features in the intermediate layer
        sequence_length (int, optional): Length of input sequence (unused but kept for API consistency)
        activation (str, optional): Activation function to use. Defaults to 'gelu'
        out_features (int, optional): Number of output features. Defaults to in_features if None

    Forward Args:
        x (torch.Tensor): Input tensor

    Returns:
        torch.Tensor: Transformed output tensor
    """

    def __init__(
        self,
        in_features,
        hidden_features,
        sequence_length=None,
        activation="gelu",
        out_features=None,
    ):
        super().__init__()
        if out_features is None:
            out_features = in_features
        self.mlp1 = torch.nn.Linear(in_features, hidden_features)
        self.activation = activations.get(activation, torch.nn.GELU)()
        self.mlp2 = torch.nn.Linear(hidden_features, out_features)

    def forward(self, x):
        x = self.mlp1(x)
        x = self.activation(x)
        x = self.mlp2(x)
        return x


class SGUMLPBlock(torch.nn.Module):
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
        sequence_length,
        activation="gelu",
        out_features=None,
    ):
        super().__init__()
        if out_features is None:
            out_features = in_features
        self.mlp1 = torch.nn.Linear(in_features, hidden_features)
        self.sgu = SpatialGatedUnit(hidden_features, sequence_length)
        self.activation = activations.get(activation, torch.nn.GELU)()
        self.mlp2 = torch.nn.Linear(hidden_features // 2, out_features)

    def forward(self, x):
        x = self.mlp1(x)
        x = self.sgu(x)
        x = self.activation(x)
        x = self.mlp2(x)
        return x


mlp_blocks = {
    "mlp": MLPBlock,
    "sgu": SGUMLPBlock,
}


class MLPMixerBlock(torch.nn.Module):
    """
    Implementation of a single MLP-Mixer block that performs token and channel mixing.

    Args:
        in_features (int): Number of input features per token
        hidden_features (int): Number of hidden features in the MLP blocks
        sequence_length (int): Number of tokens in the sequence
        activation (str, optional): Activation function to use. Defaults to 'gelu'
        mlp_block (str, optional): Type of MLP block to use ('mlp' or 'sgu'). Defaults to 'mlp'

    Forward Args:
        x (torch.Tensor): Input tensor of shape [batch_size, sequence_length, in_features]

    Returns:
        torch.Tensor: Processed tensor after token and channel mixing
    """

    def __init__(
        self,
        in_features,
        hidden_features,
        sequence_length,
        activation="gelu",
        mlp_block="mlp",
    ):
        super().__init__()
        mlp_block = mlp_blocks.get(mlp_block, MLPBlock)

        self.layer_norm = torch.nn.LayerNorm([in_features])
        self.channel_mixer = mlp_block(
            in_features=in_features,
            hidden_features=hidden_features,
            sequence_length=sequence_length,
            activation=activation,
        )
        self.token_mixer = mlp_block(
            in_features=sequence_length,
            hidden_features=hidden_features,
            sequence_length=in_features,
            activation=activation,
        )

    def forward(self, x):
        residual = x

        x = self.layer_norm(x).transpose(1, 2)
        x = self.token_mixer(x).transpose(1, 2)

        residual = residual + x

        x = self.layer_norm(residual)
        x = self.channel_mixer(x)
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
    Implementation of the MLP-Mixer architecture for image processing.

    Args:
        image_dimensions (tuple): Tuple of (height, width, channels) for input images
        patch_size (int): Size of image patches
        token_features (int): Number of features per token
        mixer_mlp_hidden_features (int): Number of hidden features in mixer MLP blocks
        num_blocks (int, optional): Number of mixer blocks to use. Defaults to 1
        activation (str, optional): Activation function to use. Defaults to 'gelu'

    Forward Args:
        x (torch.Tensor): Input tensor of shape [batch_size, height, width, channels]

    Returns:
        torch.Tensor: Processed tensor of shape [batch_size, n_tokens, token_features]
    """

    def __init__(
        self,
        image_dimensions,
        patch_size: int,
        token_features: int,
        mixer_mlp_hidden_features: int,
        num_blocks=1,
        activation="gelu",
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
                    hidden_features=mixer_mlp_hidden_features,
                    sequence_length=self.n_tokens,
                    activation=activation,
                )
            )

    def forward(self, x):
        b = x.shape[0]
        x = self.token_embedding(x.moveaxis(-1, 1))
        x = x.reshape(b, self.n_channels, self.n_tokens).transpose(1, 2)
        for block in self.mixer_blocks:
            x = block(x)
        return x


class SGUMLPMixer(torch.nn.Module):
    """
    Implementation of MLP-Mixer with Spatial Gated Units and depthwise convolutions.

    Args:
        image_dimensions (tuple): Tuple of (height, width, channels) for input images
        patch_size (int): Size of image patches
        token_features (int): Number of features per token
        mixer_mlp_hidden_features (int): Number of hidden features in mixer MLP blocks
        dwc_kernels (tuple, optional): Kernel sizes for depthwise convolutions. Defaults to (1, 3, 5)
        num_blocks (int, optional): Number of mixer blocks to use. Defaults to 1
        activation (str, optional): Activation function to use. Defaults to 'gelu'

    Forward Args:
        x (torch.Tensor): Input tensor of shape [batch_size, height, width, channels] corresponding to image_dimensions.

    Returns:
        torch.Tensor: Processed tensor of shape [batch_size, n_tokens, token_features]
    """

    def __init__(
        self,
        image_dimensions,
        patch_size: int,
        token_features: int,
        mixer_mlp_hidden_features: int,
        dwc_kernels=(1, 3, 5),
        num_blocks=1,
        activation="gelu",
    ):
        super().__init__()
        self.image_dimensions = image_dimensions
        h, w, c = image_dimensions
        self.n_tokens = calc_n_tokens(h, w, patch_size)
        self.patch_size = patch_size
        self.n_channels = token_features

        self.dwc = ParallelDepthwiseConv2d(c, dwc_kernels)
        self.token_embedding = torch.nn.Conv2d(
            c, token_features, kernel_size=patch_size, stride=patch_size
        )
        self.mixer_blocks = torch.nn.ModuleList()
        for _ in range(num_blocks):
            self.mixer_blocks.append(
                MLPMixerBlock(
                    in_features=token_features,
                    hidden_features=mixer_mlp_hidden_features,
                    sequence_length=self.n_tokens,
                    activation=activation,
                    mlp_block="sgu",
                )
            )

    def forward(self, x):
        b = x.shape[0]
        x = x.moveaxis(-1, 1)
        x = self.dwc(x)
        x = self.token_embedding(x)
        x = x.reshape(b, self.n_channels, self.n_tokens).transpose(1, 2)
        for block in self.mixer_blocks:
            x = block(x)
        return x
