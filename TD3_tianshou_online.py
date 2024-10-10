#!/usr/bin/env python3
import datetime
import os
import pprint

import numpy as np
import torch

from torch.utils.tensorboard import SummaryWriter
from tianshou.data import Collector, CollectStats, ReplayBuffer, VectorReplayBuffer
from tianshou.exploration import GaussianNoise
from tianshou.highlevel.logger import LoggerFactoryDefault
from tianshou.policy import TD3Policy
from tianshou.policy.base import BasePolicy
from tianshou.trainer import OffpolicyTrainer
from tianshou.utils.net.common import Net
from tianshou.utils.net.continuous import Actor, Critic
from tianshou.env import DummyVectorEnv
from tianshou.utils import TensorboardLogger

import gymnasium as gym


def test_td3() -> None:
    
    env = gym.make('LunarLanderContinuous-v2')

    train_envs = DummyVectorEnv([lambda: gym.make('LunarLanderContinuous-v2') for _ in range(10)])
    test_envs = DummyVectorEnv([lambda: gym.make('LunarLanderContinuous-v2') for _ in range(10)])
    state_shape = env.observation_space.shape or env.observation_space.n
    action_shape = env.action_space.shape or env.action_space.n
    max_action = env.action_space.high[0]
    # exploration_noise = 
    # policy_noise = 
    # noise_clip = 
    print("Observations shape:", state_shape)
    print("Actions shape:", action_shape)
    print("Action range:", np.min(env.action_space.low), np.max(env.action_space.high))
    # seed
    seed = 42
    np.random.seed(seed)
    torch.manual_seed(seed)

    # model
    hidden_sizes = ()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    net_a = Net(state_shape=state_shape, hidden_sizes=hidden_sizes, device=device)
    actor = Actor(net_a, action_shape, max_action=max_action, device=device).to(
        device,
    )
    actor_optim = torch.optim.Adam(actor.parameters(), lr=1e-3)
    net_c1 = Net(
        state_shape=state_shape,
        action_shape=action_shape,
        hidden_sizes=hidden_sizes,
        concat=True,
        device=device,
    )
    net_c2 = Net(
        state_shape=state_shape,
        action_shape=action_shape,
        hidden_sizes=hidden_sizes,
        concat=True,
        device=device,
    )
    critic1 = Critic(net_c1, device=device).to(device)
    critic1_optim = torch.optim.Adam(critic1.parameters(), lr=1e-3)
    critic2 = Critic(net_c2, device=device).to(device)
    critic2_optim = torch.optim.Adam(critic2.parameters(), lr=1e-3)

    policy: TD3Policy = TD3Policy(
        actor=actor,
        actor_optim=actor_optim,
        critic=critic1,
        critic_optim=critic1_optim,
        critic2=critic2,
        critic2_optim=critic2_optim,
        tau=0.005,
        gamma=0.99,
        exploration_noise=GaussianNoise(sigma=0.1),
        policy_noise=0.2,
        update_actor_freq=2,
        noise_clip=0.5,
        estimation_step=1,
        action_space=env.action_space,
        action_scaling=True
    )


    # collector
    buffer = VectorReplayBuffer(10000, 10)
    train_collector = Collector(policy, train_envs, buffer, exploration_noise=True)
    test_collector = Collector(policy, test_envs)
    train_collector.reset()
    train_collector.collect(n_step=1000, random=True)

    # log
    # now = datetime.datetime.now().strftime("%y%m%d-%H%M%S")
    # algo_name = "td3"
    # log_name = os.path.join("test", algo_name, str(seed), now)
    # log_path = os.path.join("log", log_name)

    # logger
    writer = SummaryWriter(log_dir='log/td3_lunar_lander')
    logger = TensorboardLogger(writer)

    def save_best_fn(policy: BasePolicy) -> None:
        torch.save(policy.state_dict(), 'best_policy.pth')
    result = OffpolicyTrainer(
        policy=policy,
        train_collector=train_collector,
        test_collector=test_collector,
        max_epoch=100,
        step_per_epoch=10000,
        step_per_collect=10,
        episode_per_test=10,
        batch_size=1000,
        save_best_fn=save_best_fn,
        logger=logger,
        update_per_step=0.1,
        test_in_train=False,
        ).run()
    pprint.pprint(result)

    # Let's watch its performance!
    test_envs.seed(seed)
    test_collector.reset()
    collector_stats = test_collector.collect(n_episode = 1, render = True)
    print(collector_stats)


if __name__ == "__main__":
    test_td3()