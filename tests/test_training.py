import torch
import lightning

from src.models import LitSGUMLPMixer, Classifier


def test_model_training(dataset, sgumlpmixer_args, adamw_args):
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
    model = LitSGUMLPMixer(model_params=sgu_args, optimizer_params=adamw_args)
    trainer = lightning.Trainer(accelerator="auto", min_epochs=1, max_epochs=1)
    trainer.fit(model, train_dataloaders=dl)
