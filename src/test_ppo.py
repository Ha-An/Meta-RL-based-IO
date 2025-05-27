import GymWrapper as gw
import os
import time
import multiprocessing
import math
import torch
from GymWrapper import GymInterface 
from PPO import PPOAgent
from config_RL import *
from torch.utils.tensorboard import SummaryWriter
from inner_Model import *
main_writer = SummaryWriter(log_dir=TENSORFLOW_LOGS)


total_episodes = N_EPISODES  
episode_counter = 0
env_main = GymInterface()
model = inner_model(env_main, 10, 3e-4, 20, 10)
model.main_module()
print(model.memory)