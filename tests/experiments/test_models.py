import lightning
import torch

from experiments.models import LitSGUMLPMixer


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


def test_model_lit_model_training(dataset, sgumlpmixer_args, adamw_args, tmp_path):
    x, y, k = dataset
    b, p, p, c = x.shape

    dl = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(x, y),
        batch_size=b // 4,
        num_workers=1,
        shuffle=True,
    )

    sgu_args = dict(
        input_dimensions=(p, p, c),
        **sgumlpmixer_args,
        num_classes=k,
    )

    checkpoint_cb = lightning.pytorch.callbacks.ModelCheckpoint(
        dirpath=tmp_path,
        filename="sgu_model",
    )

    model = LitSGUMLPMixer(model_params=sgu_args, optimizer_params=adamw_args)
    trainer = lightning.Trainer(
        accelerator="auto",
        min_epochs=1,
        max_epochs=1,
        default_root_dir=tmp_path,
        callbacks=[checkpoint_cb],
    )
    trainer.fit(model, train_dataloaders=dl)
    del model
    model = LitSGUMLPMixer.load_from_checkpoint(checkpoint_cb.best_model_path)
    model.to("cpu")
    model(x)
