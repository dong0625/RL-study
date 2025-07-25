import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter

from datetime import datetime

import parser
from model.base import get_model
from env.env import get_env

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

current_time = datetime.now().strftime('%m%d-%H%M%S')
writer = SummaryWriter(f'runs/{current_time}')

def train(model : nn.Module, optimizer : torch.optim.Optimizer, env, n_epoch):
    model.train()

    for ei in range(n_epoch):
        next_state = env.reset()
        total_loss = 0
        ti = 0
        ret = 0

        done = False
        while not done:
            state = next_state
            action = model.get_action(state)

            next_state, reward, terminated, trucated, _ = env.step(action)
            done = terminated or trucated
            frame = (state, action, reward, next_state, terminated)

            loss = model.get_loss(*frame)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            ti += 1
            ret += reward

        writer.add_scalar('Train/Mean_Loss', total_loss / ti, ei)
        writer.add_scalar('Train/Return', ret, ei)
        pt = f"weights/DQN_state_dict_{ei}.pt"
        torch.save(model.state_dict(), pt)

def test(model : nn.Module, env, n_epoch):
    model.eval()
    
    for ei in range(n_epoch):
        next_state = env.reset()
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

    env = get_env(config.env.name, config)
    model = get_model(config.model.name, config, env)

    if config.load_state or config.test_mod:
        state = torch.load(f"weights/DQN_state_dict_{config.load_number}.pt", DEVICE)
        model.load_state_dict(state)
    
    try:
        if config.test_mode:
            test(model, env, config.test.n_epochs)
        else:
            optimizer = torch.optim.Adam(model.parameters(), config.learning_rate)
            train(model, optimizer, env, config.train.n_epoch)
    
    finally:
        env.close()