import argparse
import numpy as np
import cv2 as cv
import torch
import gymnasium
from pathlib import Path

# Importa le classi del tuo progetto
from utils import load_cfg_file, DoomMap, ObservationTransformation, DoomNames
from dd_wrapper import Wrapper_vz_basic
from network.worldModel.dreamer import Dreamer
from network.controller.Policy import DreamerAgent

@torch.no_grad()
def main():
    # 1. Configurazione degli argomenti da linea di comando
    parser = argparse.ArgumentParser(description="Esegue agenti Dreamer in ambienti VizDoom e registra l'immaginazione")
    parser.add_argument("--envs", nargs="+", required=True, help="Lista di ambienti VizDoom da eseguire (es. VizdoomBasic-v0)")
    parser.add_argument("--dreamer_ckpt", type=str, required=True, help="Path al checkpoint del modello Dreamer")
    parser.add_argument("--policy_ckpt", type=str, required=True, help="Path al checkpoint della Policy (DreamerAgent)")
    parser.add_argument("--out_dir", type=str, default="./results", help="Cartella dove salvare i video di log")
    parser.add_argument("--fps", type=int, default=15, help="FPS del video di output")
    args = parser.parse_args()

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Utilizzando il device: {device}")

    # 2. Caricamento dei Modelli
    print(f"[INFO] Caricamento del World Model da {args.dreamer_ckpt}...")
    # NOTA: Assicurati che encoder e decoder siano gestiti correttamente nel tuo load_from_checkpoint
    rssm_model = Dreamer.load_from_checkpoint(args.dreamer_ckpt).to(device)
    rssm_model.eval()

    print(f"[INFO] Caricamento della Policy da {args.policy_ckpt}...")
    agent = DreamerAgent.load_from_checkpoint(args.policy_ckpt, dreamer_model=rssm_model).to(device)
    agent.eval()

    # Creazione cartella di output
    out_path = Path(args.out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # Inizializza preprocessore (es. ridimensionamento e conversione tensore)
    preprocess = ObservationTransformation(ft='pt')
    
    # 3. Ciclo sugli Ambienti
    for env_id in args.envs:
        print(f"\n{'='*50}\n[AVVIO AMBIENTE] {env_id}\n{'='*50}")
        
        # Inizializzazione Ambiente VizDoom
        cfg_path = load_cfg_file(env_id)
        env_raw = gymnasium.make(
            env_id,
            render_mode="rgb_array", # Usiamo rgb_array per catturare i frame e gestirli con OpenCV
            frame_skip=1,
            max_buttons_pressed=1,
            max_episode_steps=500,
            scenario_config_file=str(cfg_path)
        )
        env = Wrapper_vz_basic(env_raw, distance_reward_scale=0.05)[cite: 5]
        mapper = DoomMap(env)[cite: 6]

        # Reset dell'ambiente e stato dell'agente
        game, _ = env.reset()[cite: 6]
        obs = game["screen"][cite: 6]
        game_info = game["gamevariables"][cite: 6]
        
        # Inizializzazione della memoria (RSSM State)
        rssm_state = agent.rssm.init_state(1, device)[cite: 2]
        
        act = mapper.action_local2global(0)[cite: 3]
        local_action_hot = [v for v in act.values()][cite: 3]
        
        done = False
        step_idx = 0
        total_reward = 0.0

        # Setup Video Writer
        # Diamo per scontato che l'immagine di resize sia 64x64. Due immagini affiancate = 128x64.
        frame_h, frame_w = 64, 128 
        video_filename = str(out_path / f"session_{env_id}.mp4")
        fourcc = cv.VideoWriter_fourcc(*'mp4v')
        video_writer = cv.VideoWriter(video_filename, fourcc, args.fps, (frame_w, frame_h))

        print(f"[INFO] Inizio simulazione. Video di log verrà salvato in: {video_filename}")

        # 4. Ciclo di Gioco (Rollout)
        while not done:
            # Pre-processamento dell'osservazione
            obs_cropped = obs[100:, 54:267] if obs.shape[0] > 100 else obs # Crop se necessario[cite: 6]
            obs_resized = cv.resize(obs_cropped, (64, 64), interpolation=cv.INTER_NEAREST)[cite: 6]
            
            # Preparazione tensori per l'agente
            obs_t = preprocess(obs_resized).to(device).unsqueeze(0) # [1, C, H, W][cite: 3]
            att = mapper.stats_local2global(game_info)[cite: 3]
            att_t = torch.tensor(list(att.values()), dtype=torch.float32).unsqueeze(0).to(device)[cite: 3]
            prev_act_t = torch.tensor(local_action_hot, dtype=torch.float32).unsqueeze(0).to(device)[cite: 3]
            mask_actions_t = torch.from_numpy(mapper.act_mask).unsqueeze(0).to(device) if hasattr(mapper, 'act_mask') else None[cite: 3]

            # --- ESECUZIONE DELLA POLICY E IMMAGINAZIONE ---
            # Calcolo stato interno e azione
            embed_input = agent.dreamer_model.encode_obs(obs_t)[cite: 2]
            user_att_enc = agent.dreamer_model.encode_att(att_t)[cite: 2]
            a_t_enc = agent.dreamer_model.encode_action(prev_act_t)[cite: 2]
            
            rssm_state, _, _ = agent.rssm.observe_step(rssm_state, a_t_enc, embed_input, user_att_enc)[cite: 2]
            z_flat = rssm_state.z.flatten(1)[cite: 2]
            s = agent.dreamer_model.encode_state(rssm_state.h, z_flat)[cite: 2]
            
            action_logits, _, _ = agent.actor(s, mask_actions_t)[cite: 2]
            action_numpy = action_logits.squeeze(0).cpu().numpy().astype(np.uint8)[cite: 2]
            
            # Ottieni il frame immaginato (Decodifica dello stato 's')
            imagined_obs_t = agent.dreamer_model.predict_obs(s)[cite: 1]
            # Converti in formato numpy immagine [H, W, C]
            imagined_frame = imagined_obs_t.squeeze(0).permute(1, 2, 0).cpu().numpy()
            imagined_frame = (imagined_frame * 255.0).astype(np.uint8)
            imagined_frame = cv.cvtColor(imagined_frame, cv.COLOR_RGB2BGR)

            # Preparazione del frame reale per la visualizzazione
            real_frame = cv.cvtColor(obs_resized, cv.COLOR_RGB2BGR)
            
            # Concatena i due video (Reale a Sinistra | Immaginato a Destra)
            combined_frame = cv.hconcat([real_frame, imagined_frame])
            
            # Mostra a schermo e scrivi sul file video
            cv.imshow(f"Dreamer Session - {env_id} (Left: Real | Right: Imagined)", cv.resize(combined_frame, (512, 256), interpolation=cv.INTER_NEAREST))
            video_writer.write(combined_frame)
            cv.waitKey(1)

            # Esecuzione dell'azione nell'ambiente VizDoom
            global_action_idx = np.argmax(action_numpy, axis=-1).item()[cite: 3]
            selected_button_name = DoomNames.buttons_names[global_action_idx][cite: 3]
            local_action_idx = mapper._gym_action_names.index(selected_button_name) if selected_button_name in mapper._gym_action_names else 0[cite: 3]
            
            local_action_hot = action_numpy # Aggiorna prev action per il prossimo step

            game_next, reward, terminated, truncated, _ = env.step(local_action_idx)[cite: 3]
            done = terminated or truncated[cite: 3]
            total_reward += reward

            # Stampa le informazioni a terminale
            print(f"Step: {step_idx:03d} | Azione: {selected_button_name:15s} | Reward Step: {reward:+.2f} | Info Giocatore: {att}")

            # Aggiornamento osservazioni
            if not done:
                obs = game_next["screen"][cite: 6]
                game_info = game_next["gamevariables"][cite: 6]
            
            step_idx += 1

        # Chiusura e pulizia ambiente
        print(f"[FINE] Ambiente {env_id} terminato. Reward Totale: {total_reward:.2f}. Step Totali: {step_idx}")
        video_writer.release()
        env.close()[cite: 3]
        
    cv.destroyAllWindows()
    print("[COMPLETATO] Tutte le sessioni sono state processate ed i video salvati.")

if __name__ == "__main__":
    main()
