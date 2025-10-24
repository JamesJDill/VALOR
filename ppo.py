import torch
import torch.nn as nn
import torch.nn.functional as F

from dataclasses import dataclass


@dataclass
class PPOConfig:
    obs_dim: int
    act_dim: int
    lr: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_ratio: float = 0.2
    entropy_coef: float = 0.002
    value_coef: float = 0.5
    max_grad_norm: float = 0.5
    update_epochs: int = 10
    minibatch_size: int = 2048
    device: str = "cuda"


def compute_gae(rewards, values, dones, gamma, lam):
    T, N = rewards.shape
    adv = torch.zeros_like(rewards)
    last = torch.zeros(N, device=rewards.device)
    next_v = torch.zeros(N, device=rewards.device)
    next_d = torch.zeros(N, device=rewards.device)
    
    for t in reversed(range(T)):
        delta = rewards[t] + gamma * next_v * (1.0 - next_d) - values[t]
        last = delta + gamma * lam * (1.0 - next_d) * last
        
        adv[t] = last
        next_v = values[t] 
        next_d = dones[t]
        
    returns = adv + values
    return adv, returns


def ppo_update(cfg, net, batch, opt):
    obs, act, logp_old, adv, ret = [x.to(cfg.device) for x in batch]
    inds = torch.randperm(obs.size(0), device=cfg.device)
    
    for _ in range(cfg.update_epochs):
        for s in range(0, obs.size(0), cfg.minibatch_size):
            mb = inds[s:s+cfg.minibatch_size]
            
            logp, entropy = net.evaluate_actions(obs[mb], act[mb])
            ratio = (logp - logp_old[mb]).exp()
            
            s1 = ratio * adv[mb]
            s2 = torch.clamp(ratio, 1.0 - cfg.clip_ratio, 1.0 + cfg.clip_ratio) * adv[mb]
            pol_loss = -torch.min(s1, s2).mean()
            
            v_pred = net.value(obs[mb])
            v_loss = F.mse_loss(v_pred, ret[mb])
            loss = pol_loss + (cfg.value_coef * v_loss) - (cfg.entropy_coef * entropy.mean())
            
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), cfg.max_grad_norm)
            opt.step()
