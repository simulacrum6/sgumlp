import torch

from src.models import (
    MLPMixer,
    SGUMLPMixer,
    SpatialGatedUnit,
    MLPMixerBlock,
    MLPBlock,
    SGUMLPBlock,
    DepthWiseConv2d,
    ParallelDepthwiseConv2d,
    Classifier,
    LitSGUMLPMixer,
)


def test_MLPMixer(hs_image):
    mixer = MLPMixer(
        hs_image.shape[1:],
        patch_size=16,
        token_features=256,
        mixer_features_channel=768,
        mixer_features_sequence=768,
        num_blocks=2,
        activation="relu",
    )
    y = mixer(hs_image)
    out_shape = torch.tensor(y.shape)
    expected_shape = torch.tensor([hs_image.shape[0], mixer.n_tokens, mixer.n_channels])
    assert torch.all(torch.eq(out_shape, expected_shape))


def test_SGUMLPMixer(patches, sgumlpmixer_args):
    b, *patch_dimensions = patches.shape
    mixer = SGUMLPMixer(
        patch_dimensions,
        **sgumlpmixer_args,
    )
    y = mixer(patches)
    out_shape = torch.tensor(y.shape)
    expected_shape = torch.tensor([patches.shape[0], mixer.n_tokens, mixer.n_channels])
    assert y.shape == (b, mixer.n_tokens, mixer.n_channels)

    mixer = SGUMLPMixer(
        patch_dimensions,
        residual_weight=1.0,
        learnable_residual=True,
        embedding_kernel_size=8,
        **sgumlpmixer_args,
    )
    y = mixer(patches)

    n_classes = 12
    mixer = SGUMLPMixer(
        patch_dimensions,
        num_classes=n_classes,
        **sgumlpmixer_args,
    )
    y = mixer(patches)
    assert y.shape == (b, n_classes)


def test_SGU(tokens):
    b, t, c = tokens.shape
    sgu = SpatialGatedUnit(in_features=c, sequence_length=t)
    y = sgu(tokens)
    out_shape = torch.tensor(y.shape)
    expected_shape = torch.tensor([b, t, c // 2])
    assert torch.all(torch.eq(out_shape, expected_shape))


def test_MLPBlock(tokens):
    b, t, c = tokens.shape
    params = dict(
        in_features=c,
        sequence_length=t,
        hidden_features=32,
        activation="relu",
    )
    mlp = MLPBlock(**params)
    assert mlp(tokens).shape == tokens.shape

    out_features = 2048
    assert out_features != c
    mlp = MLPBlock(out_features=out_features, **params)
    assert mlp(tokens).shape == (b, t, out_features)

    mlp = SGUMLPBlock(**params)
    assert mlp(tokens).shape == tokens.shape


def test_MixerBlock(tokens):
    b, t, c = tokens.shape
    params = dict(
        in_features=c,
        sequence_length=t,
        hidden_features_channel=32,
        hidden_features_sequence=96,
        activation="relu",
    )
    mixer = MLPMixerBlock(mlp_block="mlp", **params)
    assert mixer(tokens).shape == tokens.shape
    assert type(mixer.token_mixer) == MLPBlock

    mixer = MLPMixerBlock(mlp_block="sgu", **params)
    assert mixer(tokens).shape == tokens.shape
    assert type(mixer.token_mixer) == SGUMLPBlock


def test_dwc(hs_image):
    b, c, h, w = hs_image.shape
    params = dict(in_channels=c, kernel_size=3)
    conv = DepthWiseConv2d(**params)
    assert conv(hs_image).shape == hs_image.shape

    conv = DepthWiseConv2d(num_kernels=2, **params)
    assert conv(hs_image).shape == (b, c * 2, h, w)

    conv = ParallelDepthwiseConv2d(in_channels=c, kernel_sizes=[1, 3, 8])
    assert conv(hs_image).shape == hs_image.shape


def test_Classifier(tokens):
    b, t, c = tokens.shape
    k = 8
    clf = Classifier(num_classes=k, in_features=c)
    assert clf(tokens).shape == (b, k)


def test_LitSGUMLPMixer(patches, sgumlpmixer_args, adamw_args):
    b, p, p, c = patches.shape
    k = 8

    sgu_params = dict(
        input_dimensions=(p, p, c),
        num_classes=k,
        **sgumlpmixer_args,
    )
    model = LitSGUMLPMixer(
        model_params=sgu_params, optimizer_params=adamw_args, meta_data={"foo": "bar"}
    )
    assert model(patches).shape == (b, k)
    assert model.hparams["model_params"].keys() == sgu_params.keys()
    assert model.hparams["optimizer_params"].keys() == adamw_args.keys()
    assert "foo" in model.hparams["meta_data"].keys()
