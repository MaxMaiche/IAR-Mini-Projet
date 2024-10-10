import gymnasium as gym
import torch
import numpy as np
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from tianshou.env import DummyVectorEnv
from tianshou.policy import TD3Policy
from tianshou.data import Collector, VectorReplayBuffer
from tianshou.trainer import OffpolicyTrainer
from tianshou.utils.net.common import Net
from tianshou.utils.net.continuous import Actor, Critic
from tianshou.utils import TensorboardLogger

env = gym.make('LunarLanderContinuous-v2')

train_envs = DummyVectorEnv([lambda: gym.make('LunarLanderContinuous-v2') for _ in range(10)])
test_envs = DummyVectorEnv([lambda: gym.make('LunarLanderContinuous-v2') for _ in range(10)])

seed = 42
np.random.seed(seed)
torch.manual_seed(seed)
train_envs.seed(seed)
test_envs.seed(seed)

state_shape = env.observation_space.shape
action_shape = env.action_space.shape

max_action = env.action_space.high[0]

hidden_sizes = ()
net_a = Net(state_shape, hidden_sizes=hidden_sizes, activation=nn.ReLU, device='cuda' if torch.cuda.is_available() else 'cpu')
actor = Actor(net_a, action_shape, max_action=max_action, device='cuda' if torch.cuda.is_available() else 'cpu').to('cuda' if torch.cuda.is_available() else 'cpu')
actor_optim = torch.optim.Adam(actor.parameters(), lr=1e-3)

net_c1 = Net(state_shape, action_shape, hidden_sizes=hidden_sizes, activation=nn.ReLU, device='cuda' if torch.cuda.is_available() else 'cpu')
net_c2 = Net(state_shape, action_shape, hidden_sizes=hidden_sizes, activation=nn.ReLU, device='cuda' if torch.cuda.is_available() else 'cpu')
critic1 = Critic(net_c1, device='cuda' if torch.cuda.is_available() else 'cpu').to('cuda' if torch.cuda.is_available() else 'cpu')
critic2 = Critic(net_c2, device='cuda' if torch.cuda.is_available() else 'cpu').to('cuda' if torch.cuda.is_available() else 'cpu')
critic1_optim = torch.optim.Adam(critic1.parameters(), lr=1e-3)
critic2_optim = torch.optim.Adam(critic2.parameters(), lr=1e-3)


policy = TD3Policy(actor, actor_optim, critic1, critic1_optim, env.action_space, 
                    critic2, critic2_optim,
                   tau=0.005, gamma=0.99, exploration_noise = None,
                   policy_noise=0.2, update_actor_freq=2, noise_clip=0.5, estimation_step=1, action_scaling=True)

train_collector = Collector(policy, train_envs, VectorReplayBuffer(total_size=100000, buffer_num=10))
test_collector = Collector(policy, test_envs)

writer = SummaryWriter(log_dir='log/td3_lunar_lander')
logger = TensorboardLogger(writer)

def save_best_fn(policy):
    torch.save(policy.state_dict(), 'best_policy.pth')

result = OffpolicyTrainer(
    policy, max_epoch = 100, batch_size=256, train_collector=train_collector, test_collector=test_collector, 
    step_per_epoch=1000, step_per_collect=10,
    episode_per_test=10,  update_per_step=0.1,
    test_in_train=False, save_best_fn=save_best_fn, logger=logger
)

for epoch, epoch_stat, info in result:
    print(f'Epoch {epoch}: {epoch_stat} ---- {info}')