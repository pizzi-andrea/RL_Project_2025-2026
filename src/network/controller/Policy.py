import lightning as L
import torch
from torch import Tensor, nn
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
            nn.LayerNorm(normalized_shape=feat_dim),
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
        
def lambda_return(rewards, values, continues, bootstrap, gamma=0.99, lam=0.95):
        """
        rewards: [B,H]
        values: [B,H]
        continues:[B,H] in [0,1]
        bootstrap:[B] value at last imagined step
        return: [B,H] targets
        """
        B, H = rewards.shape
        returns = torch.zeros_like(rewards)
        next_value = bootstrap
        for t in reversed(range(H)):
            next_value = rewards[:, t] + continues[:, t] * gamma * ((1 - lam) * values[:, t] + lam * next_value)
            returns[:, t] = next_value
        return returns


class Value(nn.Module):
    def __init__(self, feat_dim: int):
        super().__init__()
        hid = feat_dim * 2
        self.net = nn.Sequential(
            nn.Linear(feat_dim, hid),
            nn.SiLU(),
            nn.Linear(hid, hid),
            nn.SiLU(),
            nn.Linear(hid, 1),
        )


    def forward(self, feat):
        return self.net(feat).squeeze(-1) # [B]










class DreamerAgent(L.LightningModule):
    def __init__(self, dreamer_model:Dreamer, action_space_dim:int, horizont:int=15, burn_in:int=10):
        super().__init__()
        self.wm = dreamer_model
        self.wm.eval()
        for p in self.wm.parameters():
            p.requires_grad = False

        self.rssm = self.wm.rssm
        self.encoder = self.wm.encoder
        self.att_encoder = self.wm.att_encoder
        self.reward_head = self.wm.reward_head
        self.cont_head = self.wm.cont_head
        self.emb_dim = self.rssm.deter_dim + (self.rssm.L*self.rssm.K)
        self.action_space_dim = action_space_dim
        self.horizont = horizont
        self.burn_in = burn_in
        self.policy = Policy(self.emb_dim, self.action_space_dim)
        self.value = Value(self.emb_dim)
        self.entropy_coef = 1e-3
        self.gamma = 0.99
        self.lam = 0.95

    def configure_optimizers(self):
        return torch.optim.Adam(
            list(self.policy.parameters()) + list(self.value.parameters()),
            lr=3e-4
        )
        
        

    
    def setup(self, stage):
        
        if stage == "fit":
            pass
        if stage == "validate":
            pass
    
    def training_step(self, batch, batch_idx):
        print("batch user_att shape:", batch["user_att"].shape)
    
        obs = batch["obs"]
        actions = batch["action"] # [B,T,16] MultiBinary
        rewards_real = batch["reward"] # non serve per agent update, ma la tengo uguale
        done_real = batch["done"]
        user_att = batch["user_att"]
        mask = batch["mask"]
        
        idx_episode = batch["priority_idx"] # dim is [B]

        obs_burn = obs[:, :self.burn_in]
        act_burn = actions[:, :self.burn_in]
        att_burn = user_att[:, :self.burn_in]
        mask_burn = mask[:, :self.burn_in]

        with torch.no_grad():
            start_state = self.wm.posterior_last_state(
                obs_burn, act_burn, att_burn, mask=mask_burn
            )
        
        #-----------Imagination rollout (Bernoulli) + losses calculation --------------
        state = start_state

        rewards = []
        continues = []
        values = []
        entropies = []

        for t in range(self.horizont):
            feat = torch.cat([state.h, state.z], dim=-1)  # [B, emb_dim]

            logits = self.policy(feat)                   # [B,16]
            self.log("agent/logits_mean", logits.mean(), prog_bar=True)
            self.log("agent/probs_mean", torch.sigmoid(logits).mean(), prog_bar=True)

            probs = torch.sigmoid(logits)
            action_hard = (probs > 0.5).float()
            action = action_hard + probs - probs.detach()   # [B,16] Straight-Through Estimator solves the non-differentiability of sampling
            dist = torch.distributions.Bernoulli(probs=probs)
            entropies.append(dist.entropy().mean(dim=-1))

            with torch.no_grad():
                state = self.rssm.imagine_step(state, action, mask_t=None)
                feat2 = torch.cat([state.h, state.z], dim=-1)

                r = self.reward_head(feat2).squeeze(-1)          # [B]
                c_logits = self.cont_head(feat2).squeeze(-1)     # [B]
                cont = torch.sigmoid(c_logits)                   # [B]

            v = self.value(feat2)  # [B]

            rewards.append(r)
            continues.append(cont)
            values.append(v)

        rewards = torch.stack(rewards, dim=1)       # [B,H]
        continues = torch.stack(continues, dim=1)   # [B,H]
        values_seq = torch.stack(values, dim=1)     # [B,H]
        ent = torch.stack(entropies, dim=1).mean()  # scalar

        bootstrap = values_seq[:, -1].detach()
        targets = lambda_return(
            rewards, values_seq.detach(), continues, bootstrap,
            gamma=self.gamma, lam=self.lam
        )  # [B,H]

        value_loss = ((values_seq - targets) ** 2).mean()

        policy_loss = -targets.mean() - self.entropy_coef * ent

        loss = value_loss + policy_loss

        self.log_dict({
            "agent/loss": loss,
            "agent/value_loss": value_loss,
            "agent/policy_loss": policy_loss,
            "agent/entropy": ent,
            "agent/return_mean": targets.mean(),
        }, prog_bar=True)

        return loss



"""
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
"""    
        
