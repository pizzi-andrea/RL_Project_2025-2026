import os
from abc import ABC, abstractmethod
from pathlib import Path
from priority_functions import CuriosityReplayPriority, CountBasedPriority, AdversarialReplayPriority
from doomData import DoomDataset, EpisodicDoomDataset, collate_episodic_fn
from network.encoder.VAE import VAE
from network.RSSM.rssm import RSSM
from network.controller import Controller
from lightning.pytorch.loggers import TensorBoardLogger
from lightning import Trainer as LTrainer
from lightning.pytorch.loggers import TensorBoardLogger
from torch.utils.data import DataLoader


class Trainer(ABC):

    def __init__(self, log_directory):
        self.log_directory = Path(log_directory)
        
    
    @abstractmethod
    def fit(self, path_dataset_train, path_dataset_validation, batch_size, **train_args):
        pass

    def get_model(self):
        return self.model


class VAE_Trainer(Trainer):

    def __init__(self, log_directory, **model_params):
        
        super().__init__(log_directory)
        self.model = VAE(latent_dim=model_params["latent_dim"])
        self.logger = TensorBoardLogger(log_directory, name=False)
    
    def fit(self, path_dataset_train:str, path_dataset_validation:str, batch_size:int, epochs, **train_args):
        data_train = DataLoader(DoomDataset(path_dataset_train, resize=(240,320)), batch_size=batch_size,     shuffle=True, num_workers=os.cpu_count()-1)
        data_valid = DataLoader(DoomDataset(path_dataset_validation, resize=(240,320)), batch_size=batch_size, shuffle=False, num_workers=os.cpu_count()-1)
        trainer = LTrainer(max_epochs=epochs, logger=TensorBoardLogger("VAE_logs"))

        trainer.fit(self.model, train_dataloaders=data_train, val_dataloaders=data_valid)

        return self.model
    
class DreamerXP_Trainer(Trainer):

    def __init__(self, log_directory, **model_params):
        
        super().__init__(log_directory)
        
        self.logger = TensorBoardLogger(log_directory, name=False)
        self.vae = VAE(latent_dim=model_params["latent_dim"])
        self.vae.load_state_dict()
        self.rssm = RSSM()
        self.controller = Controller()
        
    
    def fit(self, path_dataset_train:str, path_dataset_validation:str, batch_size:int, epochs:int, **train_args):

        dataset_train = EpisodicDoomDataset(path_dataset_train, max_length=train_args["episode_length"], priority_buffer=train_args["priority_buffer_train"], preprocessor=train_args["preprocessor"])
        dataset_validation =  EpisodicDoomDataset(path_dataset_validation, max_length=train_args["episode_length"], priority_buffer=train_args["priority_buffer_validation"], preprocessor=train_args["preprocessor"])
        
        buff_train = train_args["priority_buffer_train"]
        buff_validation = train_args["priority_buffer_validation"]
        
        loader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=os.cpu_count()-1, collate_fn=collate_episodic_fn)
        loader_validation = DataLoader(dataset_validation, batch_size=batch_size, shuffle=True, num_workers=os.cpu_count()-1, collate_fn=collate_episodic_fn)
        

        epoch_idx = 0
        train = train_args["train_epochs"]
        validation = train_args["validation_epochs"]
        random_time = train_args["random_time"]

        for iter in range(1, epochs +1):
            self.rssm.fit()

            
            for iter_val in range(1,validation+1):
                if epoch_idx > random_time:
                    pass # push new trajectory in train  + train vae
                pass # push new trajectory in validation
        
        
        



            

        
        return self.model
        
        

    
    
        