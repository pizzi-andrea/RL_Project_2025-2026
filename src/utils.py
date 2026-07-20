import logging
import torch 
import numpy as np
import cv2 as cv
import os
from pathlib import Path
from lightning.pytorch.callbacks import Callback
def create_video_from_frames(input_folder, output_path, fps=30):
    """
    
    """
    images = sorted([Path(img) for img in os.listdir(input_folder) if img.endswith((".png", ".jpg", ".jpeg"))])
    images.sort(key=lambda f: int(Path(f).stem))
    if not images:
        print("Nessuna immagine trovata nella cartella.")
        return
    input_folder = Path(input_folder)

    frame = cv.imread( input_folder.joinpath(str(images[0])))
    height, width, layers = frame.shape

    
    
    fourcc = cv.VideoWriter_fourcc(*'mp4v') 
    video_writer = cv.VideoWriter(output_path, fourcc, fps, (width, height))

    print(f"Creazione video in corso: {len(images)} frame...")
    
    for image in images:
        img_path = input_folder.joinpath(str(image))
        video_writer.write(cv.imread(str(img_path)))

    video_writer.release()
    print(f"Video salvato con successo in: {str(output_path)}")



def load_cfg_file(env_name:str) -> Path:

    match env_name:
        case "VizdoomBasic-v1":
            name = "basic"
        case "VizdoomDeadlyCorridor-v1":
            name = "deadly_corridor"
        case _:
            raise ValueError("cfg not found")
        
    relative =  Path("../configs") / f"{name}.cfg"
    return relative.resolve()

def configure_logger(process_name: str):
    logging.basicConfig(
        level=logging.INFO,
       
        format=f"[%(levelname)s] [%(asctime)s] [{process_name}]: %(message)s",
        datefmt="%H:%M:%S"
    )
    logger = logging.getLogger(process_name)
    return logger


class ObservationTransformation:
    """Simple utils to process obs"""
    def __init__(self, obs_size:tuple[int, int]|None = None, ft:str='np'):
        self.obs_size = obs_size
        self.ft = ft
        
    def __call__(self, x):
        if self.obs_size:
            x = cv.resize(x, self.obs_size, interpolation=cv.INTER_NEAREST)
        
        if self.ft == 'pt':
            if len(x.shape) == 3:
                x = x[np.newaxis,:,:,:]
            
            x = torch.from_numpy(x).permute(0, 3, 1, 2).contiguous()
           
        
        elif  self.ft == 'np':
            pass
        
        return x
        
       






class StepLimitCallback(Callback):
    """callback that stop traning procedure when reached expected train steps"""
    def __init__(self, steps_per_round):
        super().__init__()
        self.steps_per_round = steps_per_round
        self.batch_count = 0

    def on_train_start(self, trainer, pl_module):
        self.batch_count = 0

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        self.batch_count += 1
        if self.batch_count >= self.steps_per_round:
            trainer.should_stop = True