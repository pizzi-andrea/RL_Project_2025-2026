import torch
from torch.utils.data import Dataset, DataLoader
from lightning import Trainer

from .rssm import DreamerWorldModel #cambiato path che dava problemi

# ---- CONFIG ----
CKPT = "VAE_ckpts/epoch=249-step=31500.ckpt"   # <-- metti il path reale
A = 6                          # <-- numero azioni
LATENT_DIM = 64
T = 8
BATCH = 2

def make_fake_sample(T=8, A=6):
    obs = torch.rand(T, 3, 240, 320)  # [T,3,240,320] in [0,1]
    actions = torch.zeros(T, A)
    actions[:, 0] = 1.0               # dummy one-hot
    rewards = torch.randn(T) * 0.1
    cont = torch.ones(T)              # mai terminal
    return {"obs": obs, "actions": actions, "rewards": rewards, "cont": cont}

class FakeSeqDataset(Dataset):
    def __len__(self): return 50
    def __getitem__(self, idx): return make_fake_sample(T=T, A=A)

def main():
    # --- crea il model con VAE ---
    wm = DreamerWorldModel(
        action_dim=A,
        vae_ckpt_path=CKPT,
        vae_latent_dim=LATENT_DIM,
        enc_dim=LATENT_DIM,      # IMPORTANTISSIMO
        freeze_vae=True,
        stoch_latents=32,
        stoch_classes=32,
    )

    # --- check rapidi: tipi e grad ---
    print("encoder type:", type(wm.encoder))
    print("decoder type:", type(wm.decoder))
    print("vae exists:", hasattr(wm, "vae"))
    print("vae any grad?:", any(p.requires_grad for p in wm.vae.parameters()))

    # --- check shape forward ---
    batch = make_fake_sample(T=T, A=A)
    obs = batch["obs"].unsqueeze(0)        # [B,T,3,240,320]
    actions = batch["actions"].unsqueeze(0)
    out = wm(obs, actions)

    print("x_hat:", out["x_hat"].shape)
    print("r_hat:", out["r_hat"].shape)
    print("c_logits:", out["c_logits"].shape)
    print("post_logits:", out["post_logits"].shape)
    print("prior_logits:", out["prior_logits"].shape)

    # --- mini training con Lightning (1 epoca) ---
    dl = DataLoader(FakeSeqDataset(), batch_size=BATCH, shuffle=True, num_workers=0)

    trainer = Trainer(
        max_epochs=1,
        accelerator="auto",
        devices="auto",
        log_every_n_steps=1,
        enable_checkpointing=False,
        enable_model_summary=True,
    )
    trainer.fit(wm, train_dataloaders=dl)

if __name__ == "__main__":
    main()
