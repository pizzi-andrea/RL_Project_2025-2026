import argparse
from trainer import VAE_Trainer

def main():
    pretrained_VAE = VAE_Trainer("VAE_log", latent_dim=128).fit("Doom4VAE/train", "Doom4VAE/validation", batch_size=64, num_epochs=100)
    

    
if __name__ == '__main__':
    main()