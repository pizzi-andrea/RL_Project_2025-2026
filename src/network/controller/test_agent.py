import lightning as L
from torch.utils.data import DataLoader

from doomData import EpisodicDoomDataset, collate_episodic_fn
from priority_functions import AdversarialReplayPriority  
from ..world.Dreamer import Dreamer
from .Policy import DreamerAgent  

# 1) carico dataset episodico (quello che già usi)
buff = AdversarialReplayPriority(500)
ds = EpisodicDoomDataset("Vizdoom/train", max_length=30, priority_buffer=buff)
dl = DataLoader(ds, batch_size=4, collate_fn=collate_episodic_fn)

# 2) carico/creo world model (qui metto il ckpt se c'è)
wm = Dreamer(
action_dim=16,
att_dim=6,
vae_ckpt_path="VAE_ckpts/epoch=249-step=31500.ckpt",  
enc_dim=1024, # lascia così se già lo usi
freeze_vae=True
)
print("wm.hparams.att_dim =", wm.hparams.att_dim)
print("wm.att_encoder[0].in_features =", wm.att_encoder[0].in_features)
print("WM encoder =", type(wm.encoder))   # oppure Dreamer.load_from_checkpoint("...ckpt...")
print("WM encoder =", type(wm.encoder)) # per vedere se sta caricando il vae o il convencoder

# 3) creo agente
agent = DreamerAgent(wm, action_space_dim=16, horizont=15, burn_in=10)

# 4) train corto (1-3 batch)
trainer = L.Trainer(
    max_steps=3,
    logger=True,
    enable_checkpointing=False,
    enable_model_summary=False,
    log_every_n_steps=1,
)
trainer.fit(agent, dl)