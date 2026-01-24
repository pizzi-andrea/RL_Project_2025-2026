import argparse
from trainer import VAE_Trainer, DreamerXP_Trainer

def main():
    pretrained_VAE = VAE_Trainer("VAE_log", latent_dim=128).fit("Doom4VAE/train", "Doom4VAE/validation", batch_size=64, num_epochs=100)
    DreamerXP_Trainer(log_directory="DreamerV3_XP").fit("Doom4RSSM/train", "VizdoomDefendCenter-v0", "Doom4RSSM/train", batch_size=32, epochs=2)
    

    
if __name__ == '__main__':
    main()