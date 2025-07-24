import torch

import gymnasium
import matplotlib.pyplot as plt

from importlib import import_module

class Env:
    def __init__(self, *args):
        self.env = self.state_space = self.action_space = self.fig = self.ax = None

    def render(self):
        if self.fig == None:
            self.fig, self.ax = plt.subplots()
        screen = self.env.render()
        self.ax.clear()
        self.ax.imshow(screen)
        self.ax.axis("off")
        plt.pause(0.01) 

    def reset(self):
        state, _ = self.env.reset()
        state = torch.tensor(state)
        return state
    
    def step(self, action):
        return self.env.step(action)
    
    def close(self):
        self.env.close()
        if self.fig != None:
            plt.close(self.fig)

def get_env(env_name, args) -> Env:
    env_module = import_module(env_name)
    env = env_module.Env(args)
    return env