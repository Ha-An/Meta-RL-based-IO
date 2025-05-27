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


BUFFER_SIZE = 100000
BATCH_SIZE = 20  
LEARNING_RATE = 3e-4
GAMMA = 0.99
CLIP_EPSILON = 0.2
UPDATE_STEPS = 10  
GAE_LAMBDA = 0.95
ENT_COEF = 0.0
VF_COEF = 0.5


class inner_model:
    def __init__(self, env, n_steps, lr, batch_size, total_steps):
        self.action_dims = [len(ACTION_SPACE) for _ in range(MAT_COUNT)]  # MultiDiscrete
        self.env = env
        self.step = n_steps
        self.lr = lr
        self.batch_szie = batch_size
        self.state_dim = len(self.env.reset())
        self.total_steps = total_steps
        self.build_model(self.env)

    def build_model(self, env):
        """
        Build and return a PPOAgent model using the environment's state dimension and MAT_COUNT.
        
        Args:
            env (GymInterface): The Gym environment instance.
            
        Returns:
            PPOAgent: A PPO agent instance initialized with proper hyperparameters.
        """
        self.model = PPOAgent(
            state_dim = self.state_dim,
            action_dims = self.action_dims,
            lr = self.lr,
            gamma = 0.99,
            clip_epsilon = CLIP_EPSILON,
            update_steps = self.step
        )


    def simulation_worker(self, model_state_dict):
        """
        Run a single episode in a worker process and return the transitions and total reward.
        
        Args:
            core_index: The index of the worker process.
            model_state_dict: The state dictionary of the main model.
            
        Returns:
            tuple: (core_index, episode_transitions, episode_reward)
        """
        
        self.model.policy.load_state_dict(model_state_dict)
        
        state = self.env.reset()
        done = False
        episode_transitions = []
        episode_reward = 0

        while not done:
            action, log_prob = self.model.select_action(state)
            next_state, reward, done, info = self.env.step(action)
            episode_transitions.append((state, action, reward, next_state, done, log_prob.item()))
            episode_reward += reward
            state = next_state

        return (episode_transitions)

    def process_transitions(self, transitions):
        """
        Combine and unpack transition data collected from a worker.
        
        Args:
            transitions (list): A list of transition lists from one or more workers.
        
        Returns:
            tuple: Separate lists for states, actions, rewards, next_states, dones, and log_probs.
        """
        states, actions, rewards, next_states, dones, log_probs = [], [], [], [], [], []
        for worker_transitions in transitions:
            for tr in worker_transitions:
                states.append(tr[0])
                actions.append(tr[1])
                rewards.append(tr[2])
                next_states.append(tr[3])
                dones.append(tr[4])
                log_probs.append(tr[5])

        return states, actions, rewards, next_states, dones, log_probs

    def worker_wrapper(self, model_state_dict):
        """
        Wrapper function for simulation_worker to unpack arguments.
        
        Args:
            args (tuple): A tuple containing (core_index, model_state_dict)
            
        Returns:
            tuple: The result of simulation_worker.
        """
        return self.simulation_worker(model_state_dict)
    
    def main_module(self):
        episode_counter = 0
        #self.model.policy.load_state_dict(torch.load(state_dict))
        start_time = time.time()
        
        while episode_counter < self.total_steps:
            model_state_dict = self.model.policy.state_dict()
            
            # Use imap_unordered to retrieve results as soon as they are ready (FIFO order)
            transitions = self.worker_wrapper(model_state_dict)
                
            states, actions, rewards, next_states, dones, log_probs = self.process_transitions([transitions])
            for j in range(len(states)):
                self.model.store_transition((states[j], actions[j], rewards[j], next_states[j], dones[j], log_probs[j]))
                
            _, self.memory = self.model.update()
            episode_counter += 1
            print(f"Completed {episode_counter} / {self.total_steps} episodes. Episode Reward:{sum(rewards)}")
        
        if SAVE_MODEL:
            model_path = os.path.join(SAVED_MODEL_PATH, SAVED_MODEL_NAME)
            torch.save(self.model.policy.state_dict(), model_path)
            print(f"{SAVED_MODEL_NAME} saved successfully")
        
        end_time = time.time()
        computation_time = (end_time - start_time) / 60
        print(f"Total computation time: {computation_time:.2f} minutes")

        return self.model.policy.state_dict()

    
