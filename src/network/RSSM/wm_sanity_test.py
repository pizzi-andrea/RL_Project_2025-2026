import os
import torch
from torch.utils.data import DataLoader
from torchvision.transforms.v2 import Compose, ToImage, ToDtype

from doomData import EpisodicDoomDataset, collate_episodic_fn
from priority_functions import AdversarialReplayPriority
from network.RSSM.rssm import DreamerWorldModel  # o dove sta davvero nel tuo progetto
from network.RSSM.wm_train import world_model_train_step  # cambia path se necessario

def main():
    torch.set_float32_matmul_precision("high")

    # ---------- PATHS ----------
    dataset_train = "Doom4RSSM/train"       # cambia se serve
    vae_ckpt_path = "src/VAE_ckpts/vae.ckpt"  # cambia se serve

    # ---------- PREPROCESS ----------
    # Importante: ToDtype(scale=True) => immagini in [0,1], coerente con BCE.
    preprocessor = Compose([ToImage(), ToDtype(dtype=torch.float32, scale=True)])

    # ---------- REPLAY BUFFER ----------
    # Buffer che userà il dataset per campionare episodi
    buff = AdversarialReplayPriority(buffer_size=500, alpha=0.7, eps=1e-2, p_max=1e6)

    # ---------- DATASET + LOADER ----------
    ds = EpisodicDoomDataset(
        base_path=dataset_train,
        max_length=30,
        priority_buffer=buff,
        preprocessor=preprocessor
    )

    loader = DataLoader(
        ds,
        batch_size=8,
        shuffle=True,
        num_workers=max(os.cpu_count() - 1, 1),
        collate_fn=collate_episodic_fn,
        pin_memory=True
    )

    # ---------- PRENDI 1 BATCH PER INFERIRE action_dim ----------
    batch = next(iter(loader))
    # action: [B,T] (pad con 0)
    actid = batch["action"]
    # action_dim = max action id + 1
    action_dim = int(actid.max().item()) + 1
    print("Inferred action_dim:", action_dim)

    # ---------- MODEL ----------
    wm = DreamerWorldModel(
        action_dim=action_dim,
        vae_ckpt_path=vae_ckpt_path,
        freeze_vae=True,
        enc_dim=128,          # metti uguale al latent_dim del tuo VAE (se 128)
        vae_latent_dim=128,   # idem
        deter_dim=1024,
        stoch_latents=32,
        stoch_classes=32,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    wm = wm.to(device)
    wm.train()

    opt = torch.optim.Adam([p for p in wm.parameters() if p.requires_grad], lr=3e-4)

    # ---------- LOOP DI SANITY ----------
    for step in range(1, 21):
        batch = next(iter(loader))

        metrics = world_model_train_step(
            wm=wm,
            batch=batch,
            optimizer=opt,
            replay_buffer=buff,     # <- così testiamo anche update priorità
            action_dim=action_dim   # <- esplicito, così non dipendi da fallback
        )

        loss = metrics["wm/loss"]
        if not (loss == loss) or abs(loss) == float("inf"):
            raise RuntimeError(f"Loss non finita a step {step}: {loss}")

        if step % 5 == 0:
            print(f"[step {step}] "
                  f"loss={metrics['wm/loss']:.4f} "
                  f"recon={metrics['wm/recon']:.4f} "
                  f"kl_dyn={metrics['wm/kl_dyn']:.4f} "
                  f"kl_rep={metrics['wm/kl_rep']:.4f} "
                  f"loss_ep={metrics['wm/loss_ep_mean']:.4f}")

    print("✅ Sanity test completato: forward + backward + PER update ok.")

if __name__ == "__main__":
    main()
