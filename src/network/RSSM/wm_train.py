import torch
import torch.nn.functional as F
from .utils import cat_kl_logits
from .utils import last_valid_from_mask

def _to_one_hot_action(action_ids: torch.Tensor, action_dim: int) -> torch.Tensor:
    # action_ids: [B,T] int
    return F.one_hot(action_ids.long(), num_classes=action_dim).float()

def world_model_train_step(wm, batch, optimizer, replay_buffer=None, action_dim: int | None = None):
    """
    wm: DreamerWorldModel
    batch: dict from collate_episodic_fn
    replay_buffer: your AdversarialReplayPriority (optional)
    action_dim: number of discrete actions (needed for one-hot)
    """
    device = next(wm.parameters()).device

    obs   = batch["obs"].to(device)     # [B,T,C,H,W]
    actid = batch["action"].to(device)  # [B,T]
    rew   = batch["reward"].to(device)  # [B,T]
    done  = batch["done"].to(device)    # [B,T]
    mask  = batch["mask"].to(device)    # [B,T]
    pidx  = batch["priority_idx"].to(device)  # [B]

    B, T = actid.shape

    # cont target = 1 - done (Dreamer-style)
    cont = 1.0 - done.float()

    # determine action_dim
    if action_dim is None:
        # safest fallback: infer from batch (works if batch contains max action)
        action_dim = int(wm.rssm.action_dim)

    actions = _to_one_hot_action(actid, action_dim)  # [B,T,A]

    # ---- forward ----
    out = wm(obs, actions)

    # ---- per-step losses (keep [B,T]!!) ----
    recon = F.binary_cross_entropy(out["x_hat"], obs, reduction="none").mean(dim=(2,3,4))    # [B,T]
    r_loss = F.mse_loss(out["r_hat"], rew, reduction="none")                                 # [B,T]
    c_loss = F.binary_cross_entropy_with_logits(out["c_logits"], cont, reduction="none")     # [B,T]

    q = out["post_logits"]   # [B,T,L,K]
    p = out["prior_logits"]  # [B,T,L,K]

    kl_dyn = cat_kl_logits(q.detach(), p).sum(dim=-1)   # [B,T]
    kl_rep = cat_kl_logits(q, p.detach()).sum(dim=-1)   # [B,T]

    # free nats PER STEP (non mean!)
    free = wm.hparams.free_nats
    kl_dyn = torch.clamp(kl_dyn, min=free)
    kl_rep = torch.clamp(kl_rep, min=free)

    # total per-step
    pred = recon + r_loss + c_loss
    loss_per_step = (
        wm.hparams.beta_pred * pred +
        wm.hparams.beta_dyn * kl_dyn +
        wm.hparams.beta_rep * kl_rep
    )  # [B,T]

    # masked scalar loss
    denom = mask.sum().clamp_min(1.0)
    loss = (loss_per_step * mask).sum() / denom

    if optimizer is not None:
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    # ---- PER update: per-episode loss = last valid step ----
    with torch.no_grad():
        loss_per_episode = last_valid_from_mask(loss_per_step, mask)  # [B]

        if replay_buffer is not None:
            replay_buffer.update(
                priority_idx=pidx.detach().cpu().numpy(),
                losses=loss_per_episode.detach().cpu().numpy()
            )

    return {
        "wm/loss": float(loss.detach().cpu()),
        "wm/loss_ep_mean": float(loss_per_episode.mean().detach().cpu()),
        "wm/recon": float((recon * mask).sum().detach().cpu() / denom.cpu()),
        "wm/r_loss": float((r_loss * mask).sum().detach().cpu() / denom.cpu()),
        "wm/c_loss": float((c_loss * mask).sum().detach().cpu() / denom.cpu()),
        "wm/kl_dyn": float((kl_dyn * mask).sum().detach().cpu() / denom.cpu()),
        "wm/kl_rep": float((kl_rep * mask).sum().detach().cpu() / denom.cpu()),
    }
