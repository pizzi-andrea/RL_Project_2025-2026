import lightning as L
from torch import Tensor, nn
import torch
from ..world.Dreamer import Dreamer
class Policy(nn.Module):
    def __init__(self, feat_dim:int, action_space_dim:int):
        super().__init__()
        

        hidden_dim = feat_dim * 2
        self.policy = nn.Sequential(
            nn.Linear(in_features=feat_dim, out_features=feat_dim*2),
            nn.LayerNorm(normalized_shape=hidden_dim),
            nn.Dropout(),
            nn.SiLU(inplace=True),

            nn.Linear(in_features=feat_dim*2, out_features=feat_dim*2),
            nn.LayerNorm(normalized_shape=hidden_dim),
            nn.Dropout(),
            nn.SiLU(inplace=True),

            nn.Linear(in_features=feat_dim*2, out_features=feat_dim),
            nn.LayerNorm(normalized_shape=hidden_dim),
            nn.Dropout(),
            nn.SiLU(inplace=True)
        )
        
        self.output = nn.Sequential(
            nn.Linear(in_features=feat_dim, out_features=action_space_dim),
        )

    def forward(self, emb:Tensor):
        action_emb = self.policy(emb)
        action = self.output(action_emb)
        return action
        


class DreamerAgent(L.LightningModule):
    def __init__(self, dreamer_model:Dreamer, action_space_dim:int, horizont:int=15, burn_in:int=10):
        self.rssm = dreamer_model.rssm.eval()
        self.encoder = dreamer_model.encoder.eval()
        self.decoder = dreamer_model.decoder.eval()
        self.emb_dim = self.rssm.deter_dim + (self.rssm.L*self.rssm.K)
        self.action_space_dim = action_space_dim
        self.horizont = horizont
        self.burn_in = burn_in
        self.policy = Policy(self.emb_dim, self.action_space_dim)
        
        

    
    def setup(self, stage):
        
        if stage == "fit":
            pass
        if stage == "validate":
            pass
    
    def training_step(self, batch, batch_idx):
        self.rssm = self.rssm.eval()
        self.encoder = self.encoder.eval()
        self.decoder = self.decoder.eval()
    
        obs = batch["obs"][:, self.burn_in]
        actions = batch["action"][:, self.burn_in]
        rewards = batch["reward"][:, self.burn_in]
        cont = batch["done"][:, self.burn_in]
        
        idx_episode = batch["priority_idx"][:, self.burn_in]
        user_att = batch["user_att"][:, self.burn_in]
        mask = batch["mask"][:, self.burn_in]

    def forward(self, obs, action, user_att, mask=None):
        B, T, C, H, W = obs.shape
        # -- encode attribute player with obs
        enc = self.encoder(obs)
        att_user = self.att_encoder(user_att)


        # -- fuse obs with user att, use simple cat fusion
        enc = torch.cat([enc, att_user], dim=-1)

        # -- store temporal informations
        
        
        rssm_state = self.rssm.init_state(B, obs.device)
        

        if self.training:
            
            # -- burn-in rssm initialization
            with torch.no_grad():
                for t in range(self.burn_in):
                    enc_t = enc[:, t]
                    action_t = action[:, t]
                    rssm_state, _ = self.rssm.observe_step(rssm_state, action_t, enc_t, mask)
                
            
            embed = torch.cat([rssm_state.h, rssm_state.z], dim=-1)
            action_logits = self.policy(embed)
            action = action_dist = torch.distributions.Bernoulli(logits=action_logits).sample()

            # -- immagination using rssm
            for _ in range(self.horizont):
                rssm_state = self.rssm.imagine_step(rssm_state, action)
                embed = torch.cat([rssm_state.h, rssm_state.z], dim=-1)
                action_logits = self.policy(embed)

                action_dist = torch.distributions.Bernoulli(logits=action_logits)
                action = action_dist.sample()
                

        else:
            pass
        
        
