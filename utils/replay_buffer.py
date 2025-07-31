import numpy as np
import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ReplayBuffer:
    def __init__(self, capacity, obs_shape, learning_starts):
        self.capacity = capacity
        self.obs_shape = obs_shape
        self.learning_starts = learning_starts
        
        self.states = np.zeros((capacity, *obs_shape), dtype=np.uint8)
        self.actions = np.zeros(capacity, dtype=np.int32)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.next_states = np.zeros((capacity, *obs_shape), dtype=np.uint8)
        self.dones = np.zeros(capacity, dtype=np.bool_)
        
        self.ptr = 0
        
    def store(self, state, action, reward, next_state, done):
        idx = self.ptr % self.capacity

        self.states[idx] = state
        self.actions[idx] = action
        self.rewards[idx] = reward
        self.next_states[idx] = next_state
        self.dones[idx] = done

        self.ptr += 1

    def sample(self, batch_size):
        if self.ptr > self.learning_starts:
            return None
        indices = np.random.randint(0, len(self), size=batch_size)
        
        states = torch.tensor(self.states[indices], dtype=torch.float32, device=DEVICE)
        actions = torch.tensor(self.actions[indices], dtype=torch.int64, device=DEVICE)
        rewards = torch.tensor(self.rewards[indices], dtype=torch.float32, device=DEVICE)
        next_states = torch.tensor(self.next_states[indices], dtype=torch.float32, device=DEVICE)
        dones = torch.tensor(self.dones[indices], dtype=torch.float32, device=DEVICE)
        
        return states, actions, rewards, next_states, dones
    
    def __len__(self):
        return min(self.capacity, self.ptr)