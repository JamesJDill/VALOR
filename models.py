import torch
import torch.nn as nn
import torch.nn.functional as F


class ActorCritic(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, hidden: int=256):
        super().__init__()
        self.pi = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, act_dim)
        )
        self.v = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, 1)
        )
        # starts initial actions as slightly negative so they are less noisy
        self.log_std = nn.Parameter(torch.full((act_dim,), -0.5))

    # Tanh-squashed Gaussian helpers
    def _base_dist(self, x):
        mu = self.pi(x)
        std = self.log_std.exp().expand_as(mu)
        return torch.distributions.Normal(mu, std), mu

    @staticmethod
    def _tanh_correction_logdet(a):
        return torch.log1p(-a.pow(2) + 1e-6).sum(-1)

    @staticmethod
    def _atanh(a):
        return 0.5 * (torch.log1p(a + 1e-6) - torch.log1p(-a + 1e-6))

    def sample_action(self, obs):
        base, mu = self._base_dist(obs)
        z = base.rsample()
        a = torch.tanh(z)
        logp = base.log_prob(z).sum(-1) - self._tanh_correction_logdet(a)
        return a, logp, mu

    def evaluate_actions(self, obs, actions):
        base, _ = self._base_dist(obs)
        z = self._atanh(actions)
        logp = base.log_prob(z).sum(-1) - self._tanh_correction_logdet(actions)
        entropy = base.entropy().sum(-1)
        return logp, entropy

    def value(self, x): 
        return self.v(x).squeeze(-1)


class TrajectoryDecoder(nn.Module):
    def __init__(
        self, 
        obs_dim_wo_ctx: int,
        K: int,
        hidden=128,
        steps=11
    ):
        super().__init__()
        self.steps = steps
        self.rnn = nn.GRU(obs_dim_wo_ctx, hidden, batch_first=True, bidirectional=True)
        self.head = nn.Linear(2*hidden, K)
        
    def forward(self, traj_obs_wo_ctx):
        if traj_obs_wo_ctx.dim() == 2:
            traj_obs_wo_ctx = traj_obs_wo_ctx.unsqueeze(0)
        B, T, D = traj_obs_wo_ctx.shape
        idx = torch.linspace(0, T-1, steps=self.steps, device=traj_obs_wo_ctx.device).long()
        x = traj_obs_wo_ctx[:, idx, :]
        out, _ = self.rnn(x)
        h = out.mean(1)
        return self.head(h)
    
    def loss_and_metrics(self, traj_obs_wo_ctx, ctx_ids):
        logits = self(traj_obs_wo_ctx)
        loss = F.cross_entropy(logits, ctx_ids)
        with torch.no_grad():
            acc = (logits.argmax(-1) == ctx_ids).float().mean()
        return loss, acc, logits
