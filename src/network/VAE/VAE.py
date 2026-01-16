import os
from lightning.pytorch.utilities.types import EVAL_DATALOADERS
from torch.utils.data import DataLoader
from Car_Drive_Dataset import CDD
from torch.optim import Adam
from pathlib import Path
from torch import nn
from lightning import Trainer
from torchvision.utils import save_image
import lightning as L
import torch
import torchvision

 
NUM_W=12
torch.set_float32_matmul_precision('high')

class VAE(L.LightningModule):
    def __init__(self, dataset_path:Path, image_size:tuple[int,int, int], latent_dim:int) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels=image_size[0], out_channels=32, kernel_size=4, stride=2, padding=1),  # 64x64 -> 32x32
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2, padding=1),             # 32x32 -> 16x16
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=4, stride=2, padding=1),            # 16x16 -> 8x8
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4, stride=2, padding=1),           # 8x8 -> 4x4
            nn.ReLU()
        )

        # ---------------- Latent layers ----------------
        self.mean_predictor = nn.Linear(256*4*4, latent_dim)       # flatten encoder output
        self.variance_predictor = nn.Linear(256*4*4, latent_dim)
        self.z_project = nn.Linear(latent_dim, 256*4*4)           # per il decoder

        # ---------------- Decoder ----------------
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, 2, 1),  # 4x4 -> 8x8
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),   # 8x8 -> 16x16
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, 2, 1),    # 16x16 -> 32x32
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, 4, 2, 1),     # 32x32 -> 64x64
            nn.Sigmoid()                            # output in [0,1]
        )

        self.dataset_path = Path(dataset_path)
        self.save_hyperparameters()
        
    
    def setup(self, stage:str):
        if stage == "fit":
            self.train_set       = self.dataset_path.joinpath("train")
            self.validation_set  = self.dataset_path.joinpath("validate")
            #self.test_set        = self.dataset_path.joinpath("test").glob(".png")
        if stage == "validate":
            self.validation_set  = self.dataset_path.joinpath("validate")
        if stage == "test":
            self.test_set        = self.dataset_path.joinpath("test")

    # copy by https://github.com/kuc2477/pytorch-vae/blob/master/model.py
    def reconstruction_loss(self, x_reconstructed, x):
        return nn.BCELoss(reduction="sum")(x_reconstructed, x) / x.size(0)

    def kl_divergence_loss(self,x, mean, logvar):
        return -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp()) / x.size(0)

    def train_dataloader(self):
        return DataLoader(CDD(self.train_set), batch_size=64, shuffle=True, num_workers=NUM_W)
    
    def val_dataloader(self):
        return DataLoader(CDD(self.validation_set), batch_size=64, shuffle=False, num_workers=NUM_W)
    
    def test_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(CDD(self.test_set), batch_size=1, shuffle=False, num_workers=NUM_W)

    def training_step(self, batch, batch_idx):
        y, mean, var = self.forward(batch)
        loss = self.reconstruction_loss(y, batch) + self.kl_divergence_loss(y, mean, var)*0.001
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        y, mean, var = self.forward(batch)
        loss = self.reconstruction_loss(y, batch) + self.kl_divergence_loss(y, mean, var)*0.001

        sample_imgs = batch[:3]
        y_sample = y[:3]
        
        grid_true = torchvision.utils.make_grid(nrow=3, tensor=sample_imgs)
        grid_pred = torchvision.utils.make_grid(nrow=3, tensor=y_sample)

        grid = torch.hstack((grid_true, grid_pred))
        log_dir = self.trainer.logger.log_dir or self.trainer.default_root_dir
    
        # Costruisci il nome file
        filename = f"grid_epoch_{self.current_epoch}.png"
        save_path = os.path.join(log_dir, filename)
        
        save_image(grid, save_path)
        #self.logger.experiment.add_image('example_images', grid, 0)

        self.log_dict({'val_loss': loss})
    
    def test_step(self, batch, batch_idx):
        y, mean, var = self.forward(batch)
        loss = self.reconstruction_loss(y, batch) + self.kl_divergence_loss(y, mean, var)*0.001

        #sample_imgs = batch[:6]
        #grid = torchvision.utils.make_grid(sample_imgs)
        #grid.s

        self.log_dict({'val_loss': loss})
    
    def predict_step(self, batch, batch_idx):
        y, mean, var = self.forward(batch)
        loss = self.reconstruction_loss(y, batch) + self.kl_divergence_loss(y, mean, var)*0.001

        #sample_imgs = batch[:6]
        #grid = torchvision.utils.make_grid(sample_imgs)
        #self.logger.experiment.add_image('example_images', grid, 0)

        self.log_dict({'val_loss': loss})

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=0.001)

    
    def forward(self, x):
        encode_img = self.encoder(x)
        mean_var = encode_img.view(encode_img.size(0), -1)

        mean = self.mean_predictor(mean_var)
        var = self.variance_predictor(mean_var)

        # reparameterization trick
        std = torch.exp(0.5 * var)
        eps = torch.randn_like(std)
        if self.training:
            z = mean + std * eps
        else:
            z = mean

        
        z_j = self.z_project(z)
        z_j = z_j.view(-1, 256, 4, 4)
        return self.decoder(z_j), mean, var  # no need for extra sigmoid, already in decoder
        



if __name__ == "__main__":
    from lightning.pytorch.loggers import TensorBoardLogger
    
    vae = VAE("dataset", image_size=(3,64,64), latent_dim=128)
    trainer = Trainer(max_epochs=50, logger=TensorBoardLogger("VAE_logs"))

    trainer.fit(vae)
    
    
        

    







        
