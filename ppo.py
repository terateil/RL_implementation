import torch
from torch import nn
from torch.nn import functional as F
import gymnasium as gym
import random
import numpy as np
from torch.distributions import Categorical

class ReplayMemory():
    def __init__(self, N):
        self.N = N
        self.data = []
    
    def update(self, new_data):
        if len(self.data) == self.N:
            self.data.pop(0)
        self.data.append(new_data)
    
    def sample(self, batch_size):
        return random.sample(self.data, batch_size)


class V_network(nn.Module): #V(s) -> scalar
    def __init__(self):
        super().__init__()
        self.fc1 = nn.LazyLinear(out_features=512)
        self.fc2 = nn.Linear(in_features=512, out_features=256)
        self.fc3 = nn.Linear(in_features=256, out_features=1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class policy_network(nn.Module): #stochastic: prob. dist
    def __init__(self, A):
        super().__init__()
        self.fc1 = nn.LazyLinear(out_features=512)
        self.fc2 = nn.Linear(in_features=512, out_features=256)
        self.fc3 = nn.Linear(in_features=256, out_features=A)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        x = self.stable_softmax(x)
        return x
    
    def stable_softmax(self, x):
        max_vals, _ = torch.max(x, dim=1, keepdim=True)
        exp_x = torch.exp(x - max_vals)
        softmax_output = exp_x / torch.sum(exp_x, dim=1, keepdim=True)
        return softmax_output



def main():
    total_ep = 0
    train_end_ep = 1000
    ep_end_t = 1000
    gamma = 0.99
    lam = 0.5
    eps = 0.1
    N_actor = 8
    batch_size = 4
    env_name = "CartPole-v1"
    
    env = gym.vector.make(env_name, num_envs=N_actor)
    critic = V_network()
    A = env.action_space[0].n
    actor_old = policy_network(A)
    actor = policy_network(A)

    critic.to('cuda')
    actor_old.load_state_dict(actor.state_dict())
    actor_old.to('cuda')
    actor.to('cuda')

    critic_optimizer = torch.optim.Adam(critic.parameters())
    actor_optimizer = torch.optim.Adam(actor.parameters())

    torch.save({
    'actor_state_dict': actor.state_dict(),
    'critic_state_dict': critic.state_dict(),
    }, f'DDPG/{total_ep}.pt')  

    while total_ep<train_end_ep:
        with open("DDPG/train_log.txt", "w") as f:
            obs, _ = env.reset() #batched starting obs
            advantage = [] #GAE
            td_error = []
            obs_hist = []
            action_hist = []
            rewards = []
            cumm_reward = []
            with torch.no_grad(): #data preparation
                for t in range(ep_end_t):
                    obs_hist.append(torch.from_numpy(obs).to('cuda'))
                    action_dist = actor_old(torch.from_numpy(obs).to('cuda'))
                    action = Categorical(action_dist).sample()
                    action_hist.append(action)
                    new_obs, reward, done, _, _ = env.step(action.cpu().numpy())
                    rewards.append(torch.from_numpy(reward).to('cuda'))
                    td_error.append(torch.from_numpy(reward).to('cuda') + 
                                    torch.from_numpy(done).int().to('cuda')*gamma*(critic(torch.from_numpy(new_obs).to('cuda'))).squeeze() - 
                                    critic(torch.from_numpy(obs).to('cuda')).squeeze())
                   
                    obs = new_obs
                    if np.any(done):
                        break
                
                advantage.append(td_error[len(td_error)-1])
                cumm_reward.append(rewards[len(rewards)-1])
                for t in range(len(td_error)-2, -1, -1):
                    advantage.insert(0, td_error[t] + (lam * gamma * td_error[t+1]))
                    cumm_reward.insert(0, rewards[t] + (gamma * rewards[t+1]))

                #ready data to use rightaway
                obs_hist = torch.stack(obs_hist)
                obs_hist = obs_hist.view(-1, *obs_hist.shape[2:])
                obs_hist = obs_hist[torch.randperm(obs_hist.shape[0])].float()
                action_hist = torch.stack(action_hist)
                action_hist = action_hist.view(-1, *action_hist.shape[2:])
                action_hist = action_hist[torch.randperm(action_hist.shape[0])]
                advantage = torch.stack(advantage)
                advantage = advantage.view(-1, *advantage.shape[2:])
                advantage = advantage[torch.randperm(advantage.shape[0])].float()
                cumm_reward = torch.stack(cumm_reward)
                cumm_reward = cumm_reward.view(-1, *cumm_reward.shape[2:])
                cumm_reward = cumm_reward[torch.randperm(cumm_reward.shape[0])].float()
            
            #training
            total_sample_cnt = obs_hist.shape[0]
            for i in range(total_sample_cnt//batch_size):
                state = obs_hist[batch_size*i:batch_size*(i+1)]
                adv = advantage[batch_size*i:batch_size*(i+1)]
                action = action_hist[batch_size*i:batch_size*(i+1)]
                critic_target = cumm_reward[batch_size*i:batch_size*(i+1)]
                old_prob = actor_old(state)[torch.arange(batch_size), action]
                prob = actor(state)[torch.arange(batch_size), action]
                ratio = prob/(old_prob + 1e-10)
                L_clip = torch.minimum(ratio*adv, torch.clamp(ratio, 1-eps, 1+eps)*adv)
                actor_loss = -(L_clip.mean())
                actor_optimizer.zero_grad()
                actor_loss.backward()
                actor_optimizer.step()

                critic_loss = F.mse_loss(critic(state), critic_target)
                critic_optimizer.zero_grad()
                critic_loss.backward()
                critic_optimizer.step()

            actor_old.load_state_dict(actor.state_dict())


            score = 0
            with torch.no_grad():
                test_env = gym.make(env_name)
                obs, _ = test_env.reset() 
                test_t = 0
                while test_t < ep_end_t:
                    test_t += 1
                    action_dist = actor(torch.from_numpy(np.array([obs], dtype = np.float32)).to('cuda'))[0]
                    action = Categorical(action_dist).sample().cpu().numpy()
                    obs, reward, end, _, _ = test_env.step(action)
                    score += reward
                    if end:
                        break
                print(f"training episode: {total_ep+1}, score: {score}")
            total_ep += 1

    torch.save({
    'actor_state_dict': actor.state_dict(),
    'critic_state_dict': critic.state_dict(),
    }, f'DDPG/{total_ep}.pt')    
if __name__ == "__main__":
    main()