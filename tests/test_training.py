import torch
import lightning

from src.models import LCLFModel, Classifier


def test_model_training(dataset):
    x, y = dataset
    b, t, c = x.shape
    d_ffn = 128
    k = 10

    dl = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(x, y), batch_size=64, num_workers=1, shuffle=True
    )

    clf = Classifier(num_classes=k, in_features=d_ffn, model=torch.nn.Linear(c, d_ffn))
    model = LCLFModel(model=clf)
    trainer = lightning.Trainer(accelerator="auto", min_epochs=1, max_epochs=1)
    trainer.fit(model, train_dataloaders=dl)
