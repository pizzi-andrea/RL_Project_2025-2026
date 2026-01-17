from doom_dataset import DoomDataset
from network.encoder.VAE import VAE
from lightning.pytorch.loggers import TensorBoardLogger
from lightning import Trainer
from torch.utils.data import DataLoader

if __name__ == "__main__":
    
    
    vae = VAE(latent_dim=512)
    
    data_train = DataLoader(DoomDataset("Doom4VAE/train", resize=(240,320)), batch_size=64, shuffle=True, num_workers=12)
    data_valid = DataLoader(DoomDataset("Doom4VAE/validation", resize=(240,320)), batch_size=64, shuffle=False, num_workers=12)
    data_test = DataLoader(DoomDataset("Doom4VAE/test", resize=(240,320)), batch_size=64, shuffle=False, num_workers=12)
    trainer = Trainer(max_epochs=250, logger=TensorBoardLogger("VAE_logs"))

    trainer.fit(vae, train_dataloaders=data_train, val_dataloaders=data_valid)