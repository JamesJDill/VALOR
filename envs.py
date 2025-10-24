import numpy as np
import gymnasium as gym
from gymnasium.vector import SyncVectorEnv
from gymnasium.wrappers import TimeLimit


def make_env_thunk(env_id: str, seed: int, K: int, ep_len: int=256):
    def thunk():
        env = gym.make(env_id)
        env = TimeLimit(env, max_episode_steps=ep_len)
        env.reset(seed=seed)
        env = ContextConcatWrapper(env, K=K)
        return env
    return thunk

def make_vec_env(env_id: str, num_envs: int, seed: int, K: int, ep_len: int=256):
    thunks = [make_env_thunk(env_id, seed+i, K, ep_len) for i in range(num_envs)]
    return SyncVectorEnv(thunks)


class ContextConcatWrapper(gym.ObservationWrapper):
    def __init__(self, env: gym.Env, K: int, context_id=None):
        super().__init__(env)
        self.K = int(K)
        self.fixed_ctx = context_id
        self._one=None
        
        low  = np.concatenate([self.observation_space.low.astype(np.float32), np.zeros(self.K, np.float32)])
        high = np.concatenate([self.observation_space.high.astype(np.float32), np.ones(self.K,  np.float32)])
        self.observation_space = gym.spaces.Box(low=low, high=high, dtype=np.float32)
        
    def reset(self, seed=None, options=None):
        obs, info = self.env.reset(seed=seed, options=options)
        if self.fixed_ctx is not None:
            c = int(self.fixed_ctx)
        else:
            c = int(np.random.randint(self.K))
        
        self._one = np.zeros(self.K, np.float32)
        self._one[c] = 1.0
        
        info = dict(info)
        info["context_id"] = c
        
        return np.concatenate([obs.astype(np.float32), self._one]), info
    
    def observation(self, observation):
        return np.concatenate([observation.astype(np.float32), self._one])
