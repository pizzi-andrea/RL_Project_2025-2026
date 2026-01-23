import torch
from torch.utils.data import Dataset, DataLoader
from lightning import Trainer
from rssm import DreamerWorldModel
#batch casuale di prova per testare il training

def make_fake_batch(B=8, T=32, C=3, H=64, W=64, A=6):
    obs = torch.rand(B, T, C, H, W)                 # [0,1]
    actions = torch.zeros(B, T, A)
    actions[..., 0] = 1.0                           # one-hot dummy
    rewards = torch.randn(B, T) * 0.1
    cont = torch.ones(B, T)                         # all continuing
    return {"obs": obs, "actions": actions, "rewards": rewards, "cont": cont}

class FakeSeqDataset(Dataset):
    def __init__(self, n_items=200, T=32, A=6):
        self.n_items = n_items
        self.T = T
        self.A = A

    def __len__(self):
        return self.n_items

    def __getitem__(self, idx):
        # ritorna UN sample (B=1 implicitamente); il DataLoader farà batch
        b = make_fake_batch(B=1, T=self.T, A=self.A)
        return {k: v.squeeze(0) for k, v in b.items()}  # [T,...]

def main():
    A = 6
    model = DreamerWorldModel(action_dim=A)

    ds = FakeSeqDataset(n_items=100, T=32, A=A)
    dl = DataLoader(ds, batch_size=8, shuffle=True, num_workers=0)

    trainer = Trainer(
        max_epochs=2,
        accelerator="auto",
        devices="auto",
        log_every_n_steps=1,
        enable_checkpointing=False,
    )
    trainer.fit(model, train_dataloaders=dl)

if __name__ == "__main__":
    main()
