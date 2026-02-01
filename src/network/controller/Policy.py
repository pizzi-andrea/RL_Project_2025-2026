from collections import OrderedDict
from pathlib import Path
import lightning as L
import numpy as np
from torch import Tensor, nn
import torch
import torchvision
from ..world.Dreamer import Dreamer
import gymnasium
from PIL import Image
from torchvision.transforms.functional import to_tensor
from vizdoom import gymnasium_wrapper

class Policy(nn.Module):
    def __init__(self, feat_dim:int, action_space_dim:int):
        super().__init__()

        hidden_dim = feat_dim * 2
        self.policy = nn.Sequential(
            nn.Linear(in_features=feat_dim, out_features=feat_dim*2),
            nn.LayerNorm(normalized_shape=feat_dim*2),
            nn.Dropout(),
            nn.SiLU(inplace=True),

            nn.Linear(in_features=feat_dim*2, out_features=feat_dim*2),
            nn.LayerNorm(normalized_shape=feat_dim*2),
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
        


class DreamerAgent(L.LightningModule):
    def __init__(self, dreamer_model:Dreamer|str, action_space_dim:int, entropy_coef:int=0.0003,horizont:int=15, burn_in:int=10):
        super().__init__()
        self.save_hyperparameters()
        # import dreamer model or weigths
        if isinstance(dreamer_model, str):
            self.dreamer_model = Dreamer.load_from_checkpoint(dreamer_model)
        else:
            self.dreamer_model = dreamer_model
        
        # obtain RSSM model
        self.rssm = dreamer_model.rssm
        self.entropy_coef = entropy_coef
        
        self.validation_env = gymnasium.make("VizdoomBasic-v0", render_mode="human")
        self.emb_dim = self.rssm.deter_dim + (self.rssm.L*self.rssm.K)
        self.action_space_dim = action_space_dim
        self.horizont = horizont
        self.burn_in = burn_in
        self.policy = Policy(self.emb_dim, self.action_space_dim)
        self.value_estimator = nn.Sequential(
            nn.Linear(self.emb_dim, self.emb_dim * 2),
            nn.SiLU(),
            nn.Linear(self.emb_dim * 2, self.emb_dim * 2),
            nn.SiLU(),
            nn.Linear(self.emb_dim * 2, 1),
        )

        

        
    def setup(self, stage):
        
        self.dreamer_model.freeze_model()
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.001)
    
    def training_step(self, batch, batch_idx):
        
        obs     = batch["obs"][:, 0:self.burn_in]
        actions = batch["action"][:, 0:self.burn_in]
        #rewards = batch["reward"][:, self.burn_in]
        #cont = batch["done"][:, self.burn_in]
        
        #idx_episode = batch["priority_idx"][:, self.burn_in]
        user_att = batch["user_att"][:, 0:self.burn_in]
        mask     = batch["mask"][:, 0:self.burn_in]

        output = self.forward(obs, actions, user_att, mask)

        #print(f"output values{output["values"].shape}")
        
        img_t = output["imagination"][0,:]
        
        with torch.no_grad():
            img_t = self.dreamer_model.predict_obs(img_t)
            collage = torchvision.utils.make_grid(img_t, nrow=img_t.shape[0], normalize=True)
            torchvision.utils.save_image(collage, Path(self.logger.log_dir).joinpath(f"dreamer_trajectory{self.current_epoch}.png"))

        loss = self.loss_fn(rewards_pred=output["rewards"], values_pred=output["values"], continues_pred=output["continues"], entropy=output["entropy"])
        
        self.log_dict({
            "train_loss": loss.detach()
        }, on_step=False, on_epoch=True, prog_bar=True)

        return loss
    
    def _lambda_return(self, rewards, values, continues, gamma=0.99, lam=0.95):
        """
        rewards: [B,H]
        values: [B,H]
        continues:[B,H] soft
        bootstrap:[B] value at last imagined step
        return: [B,H] targets
        """
        #print(rewards.shape)
        #print(values.shape)
        #print(continues.shape)
       
        
        
        _, H = rewards.shape
        returns = torch.zeros_like(rewards)
        next_value = values[:, -1].detach()
        for t in reversed(range(H)):
            next_value = rewards[:, t] + continues[:, t] * gamma * ((1 - lam) * values[:, t] + lam * next_value)
            returns[:, t] = next_value
        return returns
    

    def loss_fn(self, rewards_pred, continues_pred, values_pred, entropy):
        td_target = self._lambda_return(rewards_pred, values_pred, continues_pred)

        values_loss = torch.nn.functional.mse_loss(values_pred, td_target)
        policy_loss = -td_target.mean() 
        entropy_loss = -self.entropy_coef * entropy.mean()
        loss = values_loss + policy_loss+ entropy_loss

        return loss

    def validation_step(self, batch, batch_idx):
        return 0
        game, _ = self.validation_env.reset()
        s, user_att = game["screen"], game["gamevariables"]
        action = np.zeros(16, dtype=np.float32)
        mean_rewards = []
        done = False
        rssm_state = None 
    
        
        
        while not done:
            s_t = to_tensor(s).to(self.device).unsqueeze(0).unsqueeze(0)
            # unsqueeze(0) due volte per ottenere [B=1, T=1, ...]
            a_t = torch.from_numpy(action).float().unsqueeze(0).unsqueeze(0).to(self.device)
            u_t = torch.from_numpy(np.ones(shape=6)).float().unsqueeze(0).unsqueeze(0).to(self.device)
            print(u_t)
            out, rssm_state = self(s_t, a_t, u_t, rssm_state)
            
            # Rimuovi B e T per l'ambiente
            action = out.squeeze(0).squeeze(0).cpu().numpy()
            
            game, reward, term, trunc, _ = self.validation_env.step(action)
            s, user_att = game["screen"], game["gamevariables"]
            mean_rewards.append(reward)
            done = term or trunc
        
        self.log("val_reward", np.mean(mean_rewards), prog_bar=True)
        return np.mean(mean_rewards)

    def forward(self, obs, action, user_att, mask=None):
        
        B, T, C, H, W = obs.shape
        # -- encode attribute player with obs
        
        # -- fuse obs with user att, use simple cat fusion
        enc = self.dreamer_model.encode(obs, user_att)

        # -- store temporal informations
        rssm_state = self.rssm.init_state(B, obs.device)
        
        reward = []
        continues = []
        values = []
        entropys = []
        if self.training:
            state_img = []
            # -- burn-in rssm initialization
            with torch.no_grad():
                for t in range(self.burn_in):
                    enc_t = enc[:, t]
                    action_t = action[:, t]
                    m = mask[:, t]
                    rssm_state, _ = self.rssm.observe_step(rssm_state, action_t, enc_t, m)
                
            
            

            embed = torch.cat([rssm_state.h, rssm_state.z], dim=-1)
            action_logits = self.policy(embed)
            img = embed.detach()

            
            state_img.append(img)
            # -- reparametrization trick
            dist = torch.distributions.Bernoulli(logits=action_logits)
            action = dist.sample()
            action = action + dist.probs - dist.probs.detach()

            entropys.append(dist.entropy().mean(-1))
            # -- immagination using rssm
            for _ in range(self.horizont):
                
                rssm_state, _ = self.rssm.imagine_step(rssm_state, action)
               
                embed = torch.cat([rssm_state.h, rssm_state.z], dim=-1)
                img = embed

                

                reward.append(self.dreamer_model.predict_reward(embed).squeeze(-1))
                cont = (self.dreamer_model.predict_continue(embed)).squeeze(-1)
                continues.append(nn.functional.sigmoid(cont))

                state_img.append(img)
                values.append(self.value_estimator(embed))
                action_logits = self.policy(embed)

                # -- reparametrization trick
                dist = torch.distributions.Bernoulli(logits=action_logits)
                action = dist.sample()
                action = action + dist.probs - dist.probs.detach()
                entropys.append(dist.entropy().mean(-1))

            
            #print("value shape:", torch.stack(values, dim = 1).squeeze(-1).shape)
            
            return {
                "rewards"   :torch.stack(reward, dim=1), 
                "continues" : torch.stack(continues, dim=1), 
                "values"    : torch.stack(values, dim = 1).squeeze(-1), 
                "entropy"   : torch.stack(entropys, dim=1),
                "imagination": torch.stack(state_img, dim=1)
            }

        else:
            actions = []
            for t in range(T):
                a_t = action[:, t]
                x_t = enc[:, t]  # enc_x
                rssm_state, _ = self.rssm.observe_step(rssm_state, a_t, x_t, mask)
                embed = torch.cat([rssm_state.h, rssm_state.z], dim=-1)
                action_logits = self.policy(embed)
                dist = torch.distributions.Bernoulli(logits=action_logits)
                action_pred = dist.sample()
                actions.append(action_pred)
            
            actions = torch.stack(actions, dim=1)
            return actions, rssm_state


                
        
        
