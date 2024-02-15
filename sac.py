import torch
from torch import nn
from torch.nn import functional as F
import gymnasium as gym
import random
import numpy as np

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


class Q_network(nn.Module): #Q(s, a) -> scalar
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


class policy_network(nn.Module): #deterministic :mu(s) -> scalar action
    def __init__(self, A, scale, bias):
        super().__init__()
        self.fc1 = nn.LazyLinear(out_features=512)
        self.fc2 = nn.Linear(in_features=512, out_features=256)
        self.fc_mean = nn.Linear(in_features=256, out_features=A)
        self.fc_std = nn.Linear(in_features=256, out_features=A)
        self.scale = torch.from_numpy(scale).to('cuda')
        self.bias = torch.from_numpy(bias).to('cuda')

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc_mean(x), 5 * (torch.tanh(self.fc_std(x)) + 1)  # From SpinUp / Denis Yarats

    def get_action(self, x):
        mean, log_std = self(x)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()  # for reparameterization trick (mean + std * N(0,1))
        y_t = torch.tanh(x_t)
        action = y_t * self.scale + self.bias
        log_prob = normal.log_prob(x_t)
        # Enforcing Action Bound
        log_prob -= torch.log(self.scale * (1 - y_t.pow(2)) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        mean = torch.tanh(mean) * self.scale + self.bias
        return action, log_prob, mean


def main():
    total_t = 0
    train_end_t = 1000000
    ep_end_t = 1000
    replay_N = 100000
    replay_fill = 50000
    gamma = 0.99
    batch_size = 32
    test_freq = 1000
    tau = 0.001
    c = 0.1
    d=10
    alpha = 0.1
    env_name = "Hopper-v4"
    
    env = gym.make(env_name)
    replay_memory = ReplayMemory(replay_N)
    critic_1_target = Q_network()
    critic_1 = Q_network()
    critic_2_target = Q_network()
    critic_2 = Q_network()
    A = env.action_space.shape[0]
    action_scale = (env.action_space.high - env.action_space.low)/2
    action_bias = (env.action_space.high + env.action_space.low)/2
    actor = policy_network(A, action_scale, action_bias)

    critic_1_target.load_state_dict(critic_1.state_dict())
    critic_2_target.load_state_dict(critic_2.state_dict())
    critic_1.to('cuda')
    critic_1_target.to('cuda')
    critic_2.to('cuda')
    critic_2_target.to('cuda')
    actor.to('cuda')
    

    critic_1_optimizer = torch.optim.Adam(critic_1.parameters())
    critic_2_optimizer = torch.optim.Adam(critic_2.parameters())
    actor_optimizer = torch.optim.Adam(actor.parameters())

    torch.save({
    'actor_state_dict': actor.state_dict(),
    'critic_1_state_dict': critic_1.state_dict(),
    'critic_2_state_dict': critic_2.state_dict()
    }, f'SAC/{total_t}.pt')  

    while total_t<train_end_t:
        with open("SAC/train_log.txt", "w") as f:
            obs, _ = env.reset()
            for t in range(ep_end_t):
                action = actor.get_action(torch.from_numpy(np.array([obs], dtype = np.float32)).to('cuda'))[0][0].detach().cpu().numpy()
                current_obs = obs
                obs, reward, done, _, _ = env.step(action)
                done = int(done)
                replay_memory.update((torch.from_numpy(np.array(current_obs, dtype=np.float32)), torch.tensor(action, dtype=torch.float32), torch.tensor(reward, dtype=torch.float32), torch.from_numpy(np.array(obs, dtype = np.float32)), torch.tensor(done, dtype = int)))
                if len(replay_memory.data) >= replay_fill:
                    data = replay_memory.sample(batch_size)
                    obs_before, a, r, obs_after, terminated = [torch.stack(tensor, dim=0).to('cuda') for tensor in zip(*data)]
                    new_action, logprob = actor.get_action(obs_after)[0:2]
                    critic_y = torch.unsqueeze(r, 1) + (1 - torch.unsqueeze(terminated, 1)) * gamma * (torch.minimum(critic_1_target(torch.cat((obs_after, new_action), dim = 1)), 
                                                                                                                     critic_2_target(torch.cat((obs_after, new_action), dim = 1))) - alpha * logprob)
                    critic_1_loss = F.mse_loss(critic_1(torch.cat((obs_before, a), dim = 1)), critic_y)
                    critic_1_optimizer.zero_grad()
                    critic_1_loss.backward(retain_graph=True)
                    critic_1_optimizer.step()
                    critic_2_loss = F.mse_loss(critic_2(torch.cat((obs_before, a), dim = 1)), critic_y)
                    critic_2_optimizer.zero_grad()
                    critic_2_loss.backward()
                    critic_2_optimizer.step()
                    
                    a, lp = actor.get_action(obs_after)[0:2]
                    actor_loss = -((torch.minimum(critic_1(torch.cat((obs_before, a), dim = 1)),
                                                  critic_2(torch.cat((obs_before, a), dim = 1))) - alpha * lp).mean()) #by deterministic policy gradient
                    actor_optimizer.zero_grad()
                    actor_loss.backward()
                    actor_optimizer.step()

                    #soft target update
                    for p1, p2 in zip(critic_1_target.parameters(), critic_1.parameters()):
                        p1.data.copy_((1-tau)*p1.data + tau*p2.data)
                    for p1, p2 in zip(critic_2_target.parameters(), critic_2.parameters()):
                        p1.data.copy_((1-tau)*p1.data + tau*p2.data)

                    total_t += 1
                    if total_t%test_freq == 0:
                        score = 0
                        with torch.no_grad():
                            test_env = gym.make(env_name)
                            obs, _ = test_env.reset() 
                            test_t = 0
                            while test_t < ep_end_t:
                                test_t += 1
                                action = actor.get_action(torch.from_numpy(np.array([obs], dtype = np.float32)).to('cuda'))[0][0].detach().cpu().numpy()
                                obs, reward, end, _, _ = test_env.step(action)
                                score += reward
                                if end:
                                    break
                            
                        print(f"train_t: {total_t}---> score: {score}")
                        print(f"train_t: {total_t}---> score: {score}", file=f)

                if done:
                    break
    torch.save({
    'actor_state_dict': actor.state_dict(),
    'critic_1_state_dict': critic_1.state_dict(),
    'critic_2_state_dict': critic_2.state_dict()
    }, f'SAC/{total_t}.pt')    
if __name__ == "__main__":
    main()