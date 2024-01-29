from DQN import *
from preprocess import *
import gymnasium as gym
import numpy as np
import random
import torch
from gymnasium.wrappers import resize_observation, gray_scale_observation, frame_stack
import tqdm

h=4
env = gym.make("CartPole-v1")
# env = gray_scale_observation.GrayScaleObservation(env)
# env = resize_observation.ResizeObservation(env, (84, 84))
# env = frame_stack.FrameStack(env, h)

A = env.action_space.n
N = 10000
E = 1000
T = 1000
end_T = 500000

batch_size = 32
gamma = 0.99
target_update_freq = 500
train_start_size = 10000
eps_decay = 1/500000


replay_memory = ReplayMemory(N)
acting_Q = Qnet_lin(A)
target_Q = Qnet_lin(A)


def train():
    total_t = 0
    eps = 1

    target_Q.load_state_dict(acting_Q.state_dict())

    acting_Q.to('cuda')
    target_Q.to('cuda')

    optimizer = torch.optim.SGD(acting_Q.parameters(), lr = 0.00025)
    
    train = False
    while total_t<end_T:
        score = 0
        obs, _ = env.reset()
        for t in range(T):
            action = None
            if (not train):
                action = random.randint(0, A-1)
            if train: #eps-greedy
                if random.random() < eps:
                    action = random.randint(0, A-1)
                else:
                    action = torch.argmax(acting_Q(torch.from_numpy(np.array([obs], dtype = np.float32)).to('cuda')), 1).item()

            current_obs = obs
            obs, reward, done, _, _ = env.step(action)
            score += reward
            done = int(done)

            replay_memory.update((torch.from_numpy(np.array(current_obs, dtype=np.float32)), torch.tensor(action, dtype=int), torch.tensor(reward), torch.from_numpy(np.array(obs, dtype = np.float32)), torch.tensor(done, dtype = int)))
            
            if len(replay_memory.data) >= train_start_size:
                train = True

            if train:
                data = replay_memory.sample(batch_size) #list of (obs, a, r, obs, done)
                obs_before, a, r, obs_after, terminated = [torch.stack(tensor, dim=0).to('cuda') for tensor in zip(*data)]
                target = r + (1 - terminated) * gamma * torch.max(target_Q(obs_after), 1)[0]
                loss = torch.nn.functional.mse_loss(acting_Q(obs_before)[torch.arange(batch_size), a], target)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_t += 1
                eps -= eps_decay
                eps = max(eps, 0.1)
                if total_t % target_update_freq==0:
                    target_Q.load_state_dict(acting_Q.state_dict())
                if train and total_t%1000 == 0:
                    test(total_t)
            if done:
                break



    torch.save({
        'model_state_dict': acting_Q.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        }, f'dqn_{total_t}.pt')    
        
    

def test(train_t):
    with torch.no_grad():
        test_env = gym.make("CartPole-v1")
        obs, _ = test_env.reset() #save last h observations(preprocessed 84*84 image)
        score = 0
        total_t = 0
        while total_t < T:
            total_t += 1
            action = torch.argmax(acting_Q(torch.from_numpy(np.array([obs], dtype = np.float32)).to('cuda')), 1).item()
            obs, reward, end, _, _ = test_env.step(action)
            score += reward
            if end:
                break
        
        print(f"train_t: {train_t}---> score: {score}")



train()   
    
