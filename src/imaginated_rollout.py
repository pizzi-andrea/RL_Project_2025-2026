from pathlib import Path
import numpy as np
import torch
import torchvision
import gymnasium
from torchvision.transforms.functional import to_tensor

from env_info import ButtonActionMapper, GamePad, build_user_att_from_env , ATT_PLAYER_NAME


@torch.no_grad()
def save_imagined_trajectory(
    env_id: str,
    agent,
    out_dir: str,
    burn_in_steps: int,
    imagine_horizon: int,
    seed: int | None = 0,
    tag: int | str = 0,
):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    env = gymnasium.make(env_id, render_mode="rgb_array", max_buttons_pressed=0)
    mapper = ButtonActionMapper.from_env(env)
    pad = GamePad()

    obs_dict, _ = env.reset(seed=seed) if seed is not None else env.reset()

    s = obs_dict["screen"]

    device = agent.device
    agent.eval()

    rssm_state = agent.rssm.init_state(batch_size=1, device=device)
    action_global = np.zeros(agent.action_space_dim, dtype=np.float32)

    for _ in range(burn_in_steps):
        obs_t = to_tensor(s).to(device)
        att_t = build_user_att_from_env(env, obs_dict["gamevariables"], device=device) #or torch.ones(len(ATT_PLAYER_NAME), device=device) as placeholder
        old_act_t = torch.from_numpy(action_global).float().to(device)

        action_global, rssm_state = agent.act(obs_t, att_t, old_act_t, rssm_state)

        pad.reset()
        for i in range(min(agent.action_space_dim, 16)):
            pad[i] = int(action_global[i])
        env_action = mapper.apply_to_env(pad)

        obs_dict, _, term, trunc, _ = env.step(env_action)
        s = obs_dict["screen"]

        if term or trunc:
            obs_dict, _ = env.reset()
            s = obs_dict["screen"]
            rssm_state = agent.rssm.init_state(batch_size=1, device=device)
            action_global = np.zeros(agent.action_space_dim, dtype=np.float32)

    feats = []
    feat = torch.cat([rssm_state.h, rssm_state.z], dim=-1)
    action_logits = agent.policy(feat)
    dist = torch.distributions.Bernoulli(logits=action_logits)
    action = dist.sample()
    action = action + dist.probs - dist.probs.detach()

    for _ in range(imagine_horizon):
        rssm_state, _ = agent.rssm.imagine_step(rssm_state, action)
        feat = torch.cat([rssm_state.h, rssm_state.z], dim=-1)
        feats.append(feat)

        action_logits = agent.policy(feat)
        dist = torch.distributions.Bernoulli(logits=action_logits)
        action = dist.sample()
        action = action + dist.probs - dist.probs.detach()

    feats = torch.stack(feats, dim=1)
    imgs_logits = agent.dreamer_model.predict_obs(feats)
    imgs = torch.sigmoid(imgs_logits)[0]

    grid = torchvision.utils.make_grid(imgs, nrow=min(imagine_horizon, 8))
    out_path = out_dir / f"imagined_{env_id}_it{tag}_burn{burn_in_steps}_H{imagine_horizon}.png"
    torchvision.utils.save_image(grid, out_path)

    env.close()
    return str(out_path)
