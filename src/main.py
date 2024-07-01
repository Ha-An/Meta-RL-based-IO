import GymWrapper as gw
import time
import HyperparamTuning as ht  # Module for hyperparameter tuning
from GymWrapper import GymInterface
from config_SimPy import *
from config_RL import *
from stable_baselines3 import DQN, DDPG, PPO
from log_SimPy import *
from log_RL import *
from copy import deepcopy

import torch
from torch import nn, optim


# Define the MAML class


class MAML:
    def __init__(self, model, lr_inner=0.01, lr_outer=0.001, k_inner=5):
        self.model = model
        self.lr_inner = lr_inner
        self.lr_outer = lr_outer
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr_outer)

    def inner_loop(self, task, data):

        # Train the model
        self.model.learn(total_timesteps=SIM_TIME * N_EPISODES)

        # 메타 파라미터 백업
        original_state_dict = deepcopy(model.policy.state_dict())

        """ Perform the inner loop of MAML on a specific task """
        for _ in range(self.num_inner_loop):
            # Train the model
            self.model.learn(total_timesteps=SIM_TIME * N_EPISODES)

            # Evaluate the adapted model
            predictions = self.model.policy(
                evaluation_data, params=fast_weights)
            valid_error = self.model.policy.loss(
                predictions, evaluation_labels)
            meta_train_loss += valid_error

            # 메타 파라미터 백업
            original_state_dict = deepcopy(model.policy.state_dict())

        return original_state_dict

    def outer_loop(self, Scenarios):
        """ Perform the outer loop of MAML over multiple tasks """
        meta_train_loss = 0.0

        for scenario in Scenarios:
            # K개 에피소드에 대해 학습
            for _ in range(N_EPISODES):
                # Train the model
                self.model.learn(total_timesteps=SIM_TIME)

            # 메타 파라미터 백업
            original_state_dict = deepcopy(model.policy.state_dict())

            # Evaluate the adapted model
            predictions = self.model.policy(
                evaluation_data, params=fast_weights)
            valid_error = self.model.policy.loss(
                predictions, evaluation_labels)
            meta_train_loss += valid_error
            # mean_reward, std_reward = gw.evaluate_model(model, env, N_EVAL_EPISODES)

        # Take the meta-learning step
        meta_train_loss /= self.meta_batch_size
        self.optimizer.zero_grad()
        meta_loss.backward()
        self.optimizer.step()

        # 모델을 원래 메타 파라미터로 복원
        model.load_state_dict(original_state_dict)

# Function to build the model based on the specified reinforcement learning algorithm


def build_model():
    if RL_ALGORITHM == "DQN":
        model = DQN("MlpPolicy", env, verbose=0)
        # model = DQN("MlpPolicy", env, learning_rate=BEST_PARAMS['learning_rate'], gamma=BEST_PARAMS['gamma'],
        #             batch_size=BEST_PARAMS['batch_size'], verbose=0)
    elif RL_ALGORITHM == "DDPG":
        model = DQN("MlpPolicy", env, verbose=0)
        # model = DDPG("MlpPolicy", env, learning_rate=BEST_PARAMS['learning_rate'], gamma=BEST_PARAMS['gamma'],
        #              batch_size=BEST_PARAMS['batch_size'], verbose=0)
    elif RL_ALGORITHM == "PPO":
        model = PPO("MlpPolicy", env, verbose=0, n_steps=SIM_TIME)
        # model = PPO("MlpPolicy", env, learning_rate=BEST_PARAMS['learning_rate'], gamma=BEST_PARAMS['gamma'],
        #             batch_size=BEST_PARAMS['batch_size'], n_steps=SIM_TIME, verbose=0)
        print(env.observation_space)
    return model


# Start timing the computation
start_time = time.time()

# Create environment
env = GymInterface()

# Initialize the model
model = build_model()

# Initialize MAML
maml = MAML(model)

# Example task structure
tasks = [
    {"train": torch.tensor(...), "validation": torch.tensor(...),
     "targets": torch.tensor(...), "val_targets": torch.tensor(...)},
    # Add more tasks
]

# Run MAML training
NUM_EPOCHS = 10

for epoch in range(NUM_EPOCHS):
    # Sample batch of scenarios

    maml.outer_loop(tasks)


# # Run hyperparameter optimization if enabled
# if OPTIMIZE_HYPERPARAMETERS:
#     ht.run_optuna()
#     # Calculate computation time and print it
#     end_time = time.time()
#     print(f"Computation time: {(end_time - start_time)/60:.2f} minutes ")
# else:
#     # Build the model
#     if LOAD_MODEL:
#         if RL_ALGORITHM == "DQN":
#             model = DQN.load(os.path.join(
#                 SAVED_MODEL_PATH, LOAD_MODEL_NAME), env=env)

#         elif RL_ALGORITHM == "DDPG":
#             model = DDPG.load(os.path.join(
#                 SAVED_MODEL_PATH, LOAD_MODEL_NAME), env=env)

#         elif RL_ALGORITHM == "PPO":
#             model = PPO.load(os.path.join(
#                 SAVED_MODEL_PATH, LOAD_MODEL_NAME), env=env)
#         print(f"{LOAD_MODEL_NAME} is loaded successfully")
#         policy_weights = model.policy.state_dict()
#         print(policy_weights.keys())
#     else:
#         model = build_model()
#         policy_weights = model.policy.state_dict()
#         print(policy_weights.keys())

#         # Train the model
#         model.learn(total_timesteps=SIM_TIME * N_EPISODES)
#         if SAVE_MODEL:
#             model.save(os.path.join(SAVED_MODEL_PATH, SAVED_MODEL_NAME))
#             print(f"{SAVED_MODEL_NAME} is saved successfully")

#         if STATE_TRAIN_EXPORT:
#             gw.export_state('TRAIN')
#     training_end_time = time.time()
#     '''
#     # Evaluate the trained model
#     mean_reward, std_reward = gw.evaluate_model(model, env, N_EVAL_EPISODES)
#     print(
#         f"Mean reward over {N_EVAL_EPISODES} episodes: {mean_reward:.2f} +/- {std_reward:.2f}")
#     # Calculate computation time and print it
#     end_time = time.time()
#     print(f"Computation time: {(end_time - start_time)/60:.2f} minutes \n",
#           f"Training time: {(training_end_time - start_time)/60:.2f} minutes \n ",
#           f"Test time:{(end_time - training_end_time)/60:.2f} minutes")
#     '''

# # Optionally render the environment
# env.render()
