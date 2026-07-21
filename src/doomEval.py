import argparse
import numpy as np
import torch
import csv
import cv2 as cv
from pathlib import Path
from math import inf

from globals import *
from utils import *
from dd_wrapper import EnviromentWrapperBuilder, DoomMap
from vizdoom import gymnasium_wrapper

from network.worldModel.dreamer import Dreamer
from network.controller.Policy import DreamerAgent

from yaml import safe_load
from network.encoder.encoders import Encoder
from network.encoder.decoders import Decoder

#argparse
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Testa un agente Dreamer già allenato su un ambiente VizDoom")
    parser.add_argument("-e", "--env", required=True, type=str, help="es. VizdoomBasic-v1")
    parser.add_argument("-n", "--episodes", required=True, type=int, help="numero di episodi da giocare")
    parser.add_argument("--dreamer_ckpt", required=True, type=str, help="path al checkpoint del world model (Dreamer)")
    parser.add_argument("--policy_ckpt", required=True, type=str, help="path al checkpoint della policy (DreamerAgent)")
    parser.add_argument("--out_dir", required=False, default="./eval_results", type=str, help="cartella dove salvare i CSV")
    parser.add_argument("--save_frames", required=False, action="store_true", help="se presente, salva anche un video mp4 in risoluzione nativa")
    parser.add_argument("--resolution_w", required=False, default=64, type=int)
    parser.add_argument("--resolution_h", required=False, default=64, type=int)
    parser.add_argument("--skip_frame", required=False, default=1, type=int)

    args = parser.parse_args()

# checkpoints loading

    #config.yaml (stesse dimensioni usate in training)
    with open("config.yaml", "r") as file:
        config = safe_load(file)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] device: {device}")

    #ricostruzione encoder/decoder con le dimensioni salvate in config
    encoder_model = Encoder(
        latent_dim=config["encoder_config"]["latent_dim"],
        model_dim=config["encoder_config"]["model_dim"]
    )
    decoder_model = Decoder(
        model_dim=config["encoder_config"]["model_dim"],
        world_model_dim=config["rssm_config"]["rssm_latent_dim"],
        input_res=(config["global_config"]["img_size_h"], config["global_config"]["img_size_l"])
    )

    #caricamento World Model DreamerAgent
    print(f"[INFO] carico Dreamer da {args.dreamer_ckpt}")
    rssm_model = Dreamer.load_from_checkpoint(
        checkpoint_path     = args.dreamer_ckpt,
        encoder             = encoder_model,
        decoder             = decoder_model,
        action_dim          = config["global_config"]["action_shape"],
        att_dim             = config["global_config"]["att_shape"],
        decode_shape        = (config["global_config"]["img_channels"], config["global_config"]["img_size_h"], config["global_config"]["img_size_l"]),
        internal_hidden_dim = config["rssm_config"]["common_latent_dim"],
        stoch_classes       = config["rssm_config"]["stochastic_classes"],
        stoch_latents       = config["rssm_config"]["stochastic_latent"],
        rssm_hidden_dim     = config["rssm_config"]["rssm_latent_dim"],
        h_dim               = config["rssm_config"]["detterministic_latent_dim"],
        buckets             = config["rssm_config"]["buckets"],
        min_buckets         = config["rssm_config"]["min_buckets"],
        max_buckets         = config["rssm_config"]["max_buckets"],
        unimix              = config["rssm_config"]["unimix"],
        weights_only=False
    ).to(device)
    rssm_model.eval()

    # caricamento Policy ()
    print(f"[INFO] carico Policy da {args.policy_ckpt}")
    agent_model = DreamerAgent.load_from_checkpoint(
        checkpoint_path     = args.policy_ckpt,
        dreamer_model       = rssm_model,
        action_space_dim    = config["global_config"]["action_shape"],
        horizon             = config["policy_config"]["horizon"],
        burn_in             = config["policy_config"]["burn_in"],
        buckets             = config["policy_config"]["buckets"],
        hidden_dim          = config["policy_config"]["hidden_dim"],
        unimix              = config["policy_config"]["unimix"],
        gamma               = config["policy_config"]["gamma"],
        min_buckets         = config["policy_config"]["min_buckets"],
        max_buckets         = config["policy_config"]["max_buckets"],
        weights_only=False
    ).to(device)
    agent_model.eval()

    # Rollout cycle
    preprocess = ObservationTransformation(ft='pt')

    env = EnviromentWrapperBuilder.get_env_from_id(
        args.env,
        gmode="rgb_array",
        skip_frame=args.skip_frame,
        resolution_w=args.resolution_w,
        resolution_h=args.resolution_h,
    )
    mapper = DoomMap(env)

    all_steps = []      # una riga per ogni singolo step (per il CSV dettagliato)
    all_episodes = []   # una riga per ogni episodio (per il CSV riassuntivo)

    for ep in range(args.episodes):
        # memoria del world model azzerata a inizio episodio: l'agente non deve qualcosa di ep precedente
        rssm_state = agent_model.dreamer_model.rssm.init_state(1, device)

        game, _ = env.reset()
        obs = game["screen"]
        att = mapper.stats_local2global(game["gamevariables"])
        act = mapper.action_local2global(0)          # azione "nulla" iniziale
        prev_action_hot = [v for v in act.values()]

        done = False
        step_idx = 0
        total_reward = 0.0

        print(f"\n[EPISODIO {ep+1}/{args.episodes}] ambiente: {args.env}")

        while not done:
            # prepara i tensori di input per la posterior
            obs_t = preprocess(obs).to(device)
            att_t = torch.tensor(list(att.values()), dtype=torch.float32).unsqueeze(0).to(device)
            prev_act_t = torch.tensor(prev_action_hot, dtype=torch.float32).unsqueeze(0).to(device)
            mask_t = torch.from_numpy(mapper.act_mask).unsqueeze(0).to(device)

            # l'agente calcola la posterior e sceglie l'azione
            action_one_hot, rssm_state = agent_model.act(obs_t, att_t, prev_act_t, rssm_state, mask_t)

            # action_one_hot è un vettore a 24 posizioni (il set GLOBALE di bottoni Doom)
            # va tradotto nell'indice LOCALE che questo specifico ambiente accetta
            global_action_idx = np.argmax(action_one_hot, axis=-1).item()
            button_name = DoomNames.buttons_names[global_action_idx]
            local_action_idx = mapper._gym_action_names.index(button_name)

            # eseguo l'azione nell'ambiente vero
            game_next, reward, terminated, truncated, _ = env.step(local_action_idx)
            done = terminated or truncated
            att_next = mapper.stats_local2global(game_next["gamevariables"])

            # registro di questo step
            all_steps.append({
                "episode": ep,
                "step": step_idx,
                "action": button_name,
                "reward": reward,
                **att_next,
            })

            total_reward += reward
            step_idx += 1

            # prepara il prossimo giro: obs/att/azione diventano quelle "vecchie" ---
            obs = game_next["screen"]
            att = att_next
            prev_action_hot = action_one_hot.tolist()

        print(f"[EPISODIO {ep+1}] fine — reward totale: {total_reward:.2f} | "
              f"lunghezza: {step_idx} step | terminated={terminated} truncated={truncated}")

        all_episodes.append({
            "episode": ep,
            "total_reward": total_reward,
            "episode_length": step_idx,
            "terminated": terminated,
            "truncated": truncated,
        })

    env.close()