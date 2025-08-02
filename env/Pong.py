import gymnasium as gym
from gymnasium.wrappers import AtariPreprocessing, FrameStackObservation

import ale_py

from env.base import Env

class Pong(Env):
    def __init__(self, config):
        super().__init__(config)
        self.env = gym.make("ALE/Pong-v5", render_mode='rgb_array')

        self.env = AtariPreprocessing(self.env, frame_skip=1)
        self.env = FrameStackObservation(self.env, stack_size=4)
        
        self.state_shape = self.env.observation_space.shape
        self.action_space = self.env.action_space.n