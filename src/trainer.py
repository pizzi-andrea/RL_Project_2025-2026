import os
from abc import ABC, abstractmethod
from pathlib import Path
from priority_functions import CuriosityReplayPriority
from doom_dataset import DoomDataset
from network.encoder.VAE import VAE
from lightning.pytorch.loggers import TensorBoardLogger
from lightning import Trainer as LTrainer
from lightning.pytorch.loggers import CSVLogger, TensorBoardLogger
from torch.utils.data import DataLoader


class Trainer(ABC):

    def __init__(self, log_directory):
        self.log_directory = Path(log_directory)
        
    
    @abstractmethod
    def fit(self, path_dataset_train, path_dataset_validation, **train_args):
        pass

    def get_model(self):
        return self.model


class VAE_Trainer(Trainer):

    def __init__(self, log_directory, **model_params):
        
        super().__init__(log_directory)
        self.model = VAE(latent_dim=model_params["latent_dim"])
        self.logger = TensorBoardLogger(log_directory, name=False)
    
    def fit(self, path_dataset_train:str, path_dataset_validation:str, **train_args):
        data_train = DataLoader(DoomDataset(path_dataset_train, resize=(240,320)), batch_size=train_args["batch_size"],      shuffle=True, num_workers=os.cpu_count()-1)
        data_valid = DataLoader(DoomDataset(path_dataset_validation, resize=(240,320)), batch_size=train_args["batch_size"], shuffle=False, num_workers=os.cpu_count()-1)
        trainer = LTrainer(max_epochs=train_args["num_epochs"], logger=TensorBoardLogger("VAE_logs"))

        trainer.fit(self.model, train_dataloaders=data_train, val_dataloaders=data_valid)

        return self.model
    
class DreamerXP_Trainer(Trainer):

    def __init__(self, log_directory, **model_params):
        
        super().__init__(log_directory)
        
        self.logger = TensorBoardLogger(log_directory, name=False)
        
    
    def fit(self, path_dataset_train:str, path_dataset_validation:str, **train_args):
        return self.model
        
        

    
    
        