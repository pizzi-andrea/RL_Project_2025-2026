import os
import torch
import pandas as pd
import argparse
import numpy as np
from PIL import Image
from pathlib import Path
from env_info import ATT_PLAYER_NAME, GLOBAL_BUTTONS
from torch.nn.functional import one_hot
from env_info import GLOBAL_BUTTONS, ATT_PLAYER_NAME

class Callback_one_hot:
    def __init__(self, num_classes:int, buttons_name:list[str]):
        self.num_classes  = num_classes
        self.buttons_name = buttons_name
    
    @staticmethod
    def callback_one_hot(x, num_classes:int, buttons_name:list[str]):
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x)
        encode = one_hot(x.long(), num_classes)
        button_dict = {b: encode[i].item() for i, b in enumerate(buttons_name)}
        return button_dict
    
    def __call__(self, x):
        return self.callback_one_hot(x, self.num_classes, self.buttons_name)
        
class Callback_multi_buttons:
    def __init__(self, button_map, button_name):
        self.button_map = button_map
        self.button_name = button_name

    @staticmethod
    def callback_multi_buttons(x, button_map, button_name):
        button_dict = {n: int(b) for b, n in zip(x, button_map)}
        for n in set(button_name) - set(button_dict.keys()):
            button_dict[n] = 0
        return button_dict
    
    def __call__(self, x):
        return self.callback_multi_buttons(x, self.button_map, self.button_name)

class callback_att_to_dict:
    def __init__(self, att_map, att_name):
        self.att_map = att_map
        self.att_name = att_name

    @staticmethod
    def callback_att_to_dict(x, att_map, att_name):
        att_dict = {n: float(b) for b, n in zip(x, att_map)}
        for n in set(att_name) - set(att_dict.keys()):
            att_dict[n] = 0.0
        return att_dict

    def __call__(self, x):
        return self.callback_att_to_dict(x, self.att_map, self.att_name)

def dict2folder(episode:list[dict], out_dir:str|Path, button_encoder=None, att_player_encoder=None, csv_name:str="episode.csv"):
    
    out_dir = Path(out_dir)
    obs_dir = out_dir.joinpath("obs")
    obs_dir.mkdir(parents=True,exist_ok=True)
    
    for idx, _ in enumerate(episode):
        
        frame_path = obs_dir.joinpath(f"{idx}.png")
        Image.fromarray(episode[idx]["obs"]).save(frame_path)
        episode[idx].pop("obs")
        episode[idx]["path_obs"] = str(frame_path)

    
        if button_encoder:
            episode[idx] = episode[idx] | button_encoder(episode[idx]["action"])
            episode[idx].pop("action")

        if att_player_encoder:
            episode[idx]  = episode[idx] | att_player_encoder(episode[idx]["att_player"])
            episode[idx].pop("att_player")
    df = pd.DataFrame(episode)
    df.to_csv(out_dir.joinpath(csv_name), index=False, sep=",")
    return episode

def pt2folder(pt_file_path: str, output_folder_name: str):
    """
    Converte un file .pt in cartella.
    - Le immagini vanno in obs/
    - 'action' viene esploso in colonne booleane (GLOBAL_BUTTONS)
    - 'user_att' viene esploso in colonne float (ATT_PLAYER_NAME)
    """
    pt_path = Path(pt_file_path)
    base_path = Path(output_folder_name)
    obs_path = base_path / "obs"

    if not pt_path.exists():
        raise FileNotFoundError(f"Il file {pt_path} non esiste.")

    os.makedirs(obs_path, exist_ok=True)

    # Caricamento su CPU
    data = torch.load(pt_path, map_location='cpu')
    
    # Dizionario che conterrà le colonne del CSV
    csv_data = {}

    # 1. GESTIONE IMMAGINI (PIL)
    if "obs" in data:
        observations = data["obs"]
        if isinstance(observations, torch.Tensor):
            observations = observations.numpy()
        
        num_frames = len(observations)
        print(f"Salvataggio di {num_frames} immagini in {obs_path}...")
        
        for i in range(num_frames):
            frame_data = observations[i]
            
            # Conversione shape [C,H,W] -> [H,W,C]
            if frame_data.ndim == 3 and frame_data.shape[0] in [1, 3, 4]: 
                frame_data = np.transpose(frame_data, (1, 2, 0))
            
            # Se grayscale [H,W,1] -> [H,W]
            if frame_data.ndim == 3 and frame_data.shape[-1] == 1:
                frame_data = frame_data.squeeze(-1)

            img = Image.fromarray(frame_data.astype(np.uint8))
            img.save(obs_path / f"frame_{i}.png")
        
        # Rimuoviamo obs per non processarlo dopo
        del data["obs"]

    # 2. GESTIONE SPECIFICA: user_att (Attributi Giocatore)
    if "user_att" in data:
        att_tensor = data["user_att"]
        if isinstance(att_tensor, torch.Tensor):
            att_tensor = att_tensor.numpy()
        
        # Controllo di sicurezza sulle dimensioni
        if att_tensor.shape[1] == len(ATT_PLAYER_NAME):
            print("Mappatura colonne user_att...")
            for i, name in enumerate(ATT_PLAYER_NAME):
                csv_data[name] = att_tensor[:, i] # Estrai la colonna i-esima
        else:
            print(f"WARNING: user_att ha shape {att_tensor.shape}, ma ci aspettavamo {len(ATT_PLAYER_NAME)} attributi. Verrà salvato come lista.")
            csv_data["user_att"] = list(att_tensor)
        
        del data["user_att"]

    # 3. GESTIONE SPECIFICA: action (Bottoni Globali)
    if "action" in data:
        act_tensor = data["action"]
        if isinstance(act_tensor, torch.Tensor):
            act_tensor = act_tensor.numpy()
            
        # Controllo di sicurezza sulle dimensioni
        if act_tensor.shape[1] == len(GLOBAL_BUTTONS):
            print("Mappatura colonne action...")
            for i, name in enumerate(GLOBAL_BUTTONS):
                csv_data[name] = act_tensor[:, i].astype(int) # 0 o 1
        else:
            print(f"WARNING: action ha shape {act_tensor.shape}, ma ci aspettavamo {len(GLOBAL_BUTTONS)} bottoni. Verrà salvato come lista.")
            csv_data["action"] = list(act_tensor)
            
        del data["action"]

    # 4. GESTIONE DATI RIMANENTI (reward, done, ids, etc.)
    # Qualsiasi altro dato nel dizionario viene gestito qui
    ref_len = None
    if csv_data:
        ref_len = len(next(iter(csv_data.values())))

    for key, val in data.items():
        if isinstance(val, torch.Tensor):
            arr = val.numpy()
        else:
            arr = np.array(val)
        
        # Gestione dimensionale per evitare l'errore di Pandas
        if arr.ndim > 1:
            if arr.shape[1] == 1:
                csv_data[key] = arr.flatten() # [N,1] -> [N]
            else:
                csv_data[key] = list(arr) # [N, M] -> Lista di array (compatibile Pandas)
        else:
            csv_data[key] = arr

    # 5. SALVATAGGIO CSV
    if csv_data:
        df = pd.DataFrame(csv_data)
        
        # Inseriamo il percorso immagine come prima colonna
        # Assumiamo che il numero di righe corrisponda al numero di frame processati o ai dati presenti
        if "obs" not in locals(): 
             # Se non c'erano obs, calcoliamo i path ipotetici o saltiamo
             pass 
        
        # Assicuriamoci che la colonna image_path abbia la lunghezza corretta
        if len(df) > 0:
            df.insert(0, "image_path", [f"obs/{i}.png" for i in range(len(df))])

        csv_file = base_path / "episode.csv"
        df.to_csv(csv_file, index=False)
        print(f"Dati tabellari salvati in {csv_file}")
    
    print(f"Finito. Cartella creata: {base_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-f", "--pt_file", required=True, type=str,
    )

    parser.add_argument(
        "-o", "--episode_name", required=False, default=".out", type=str
    )

    args = parser.parse_args()

    pt2folder(pt_file_path=args.pt_file, output_folder_name=args.episode_name)