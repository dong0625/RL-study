import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter
#tensorboard --logdir=runs

from datetime import datetime

import parser
from model.base import get_model
from env.base import get_env

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

current_time = datetime.now().strftime('%m%d-%H%M%S')
writer = SummaryWriter(f'runs/{current_time}')

def train(model : nn.Module, optimizer : torch.optim.Optimizer, env, steps):
    model.train()

    next_state, _ = env.reset()
    total_loss = ret = episode_length = 0

    for step_i in range(steps):
        state = next_state
        action = model.get_action(state)

        next_state, reward, terminated, trucated, _ = env.step(action)
        frame = (state, action, reward, next_state, done := terminated | trucated)

        loss = model.get_loss(*frame)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        ret += reward
        total_loss += loss.item()
        episode_length += 1

        if done or step_i == steps - 1:
            writer.add_scalar('Train/Loss', total_loss / episode_length, step_i)
            writer.add_scalar('Train/Reward', ret, step_i)
            next_state, _ = env.reset()
            ret = total_loss = episode_length = 0

        if step_i % 100000 == 0:
            pt = f"weights/{model.__class__.__name__}_state_dict.pt"
            torch.save(model.state_dict(), pt)

def test(model : nn.Module, env, n_episode):
    model.eval()
    
    for ei in range(n_episode):
        next_state, _ = env.reset()
        ret = 0
        
        done = False
        while not done:
            env.render()
            state = next_state
            action = model.get_action(state)

            next_state, reward, terminated, trucated, _ = env.step(action)
            ret += reward
            done = terminated or trucated

        writer.add_scalar('Test/Return', ret, ei)

if __name__ == '__main__':
    config = parser.parse()

    env = get_env(config.env)
    model = get_model(config.model, env).to(DEVICE)

    if config.load_state or config.test_mode:
        state = torch.load(f"weights/{model.__class__.__name__}_state_dict.pt", weights_only=True)
        model.load_state_dict(state)
    
    try:
        if config.test_mode:
            test(model, env, config.test.n_episode)
        else:
            optimizer = torch.optim.AdamW(model.parameters(), config.train.learning_rate)
            train(model, optimizer, env, config.train.steps)
    
    finally:
        env.close()