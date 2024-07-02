import GymWrapper as gw
import time
import HyperparamTuning as ht  # Module for hyperparameter tuning
from GymWrapper import GymInterface
from config_SimPy import *
from config_RL import *
from log_SimPy import *
from log_RL import *
import numpy as np
import gym
import torch
from torch import nn
from stable_baselines3 import PPO
from stable_baselines3.common.logger import configure
from torch.utils.tensorboard import SummaryWriter

# Hyperparameters
alpha = 0.01  # Inner loop step size (사용되지 않는 값) ->  SB3 PPO 기본 값 확인하기
beta = 0.001  # Outer loop step size ## Default: 0.001
num_scenarios = 5  # Number of full scenarios for meta-training
scenario_batch_size = 2  # Batch size for random chosen scenarios
num_inner_updates = N_EPISODES  # Number of gradient steps for adaptation
num_outer_updates = 100  # Number of outer loop updates -> meta-training iterations

# Meta-learning algorithm


class MetaLearner:
    def __init__(self, env, policy='MlpPolicy', alpha=alpha, beta=beta):
        """
        Initializes the MetaLearner with the specified environment and hyperparameters.
        """
        self.env = env
        self.policy = policy
        self.alpha = alpha
        self.beta = beta
        self.meta_model = PPO(policy, self.env, verbose=0, n_steps=SIM_TIME)
        self.logger = configure()
        self.writer = SummaryWriter(log_dir='./tensorboard_logs')

    def adapt(self, scenario, num_updates=num_inner_updates):
        """
        Adapts the meta-policy to a specific task using gradient descent.
        """
        self.env.scenario = scenario  # Set the scenario for the environment
        adapted_model = PPO(self.policy, self.env, verbose=0, n_steps=SIM_TIME)
        adapted_model.policy.load_state_dict(
            self.meta_model.policy.state_dict())
        for _ in range(num_updates):
            # Train the policy on the specific scenario
            adapted_model.learn(total_timesteps=SIM_TIME)
        return adapted_model

    def meta_update(self, scenario_models):
        """
        Performs the meta-update step by averaging gradients across scenarios.
        """
        meta_grads = []
        for scenario_model in scenario_models:
            # Retrieve gradients from the adapted policy
            grads = []
            for param in scenario_model.policy.parameters():
                grads.append(param.grad.clone())
            meta_grads.append(grads)

        # Average gradients across tasks
        mean_meta_grads = [torch.mean(torch.stack(
            meta_grads_i), dim=0) for meta_grads_i in zip(*meta_grads)]

        # Update meta-policy parameters using the outer loop learning rate
        for param, meta_grad in zip(self.meta_model.policy.parameters(), mean_meta_grads):
            param.data -= self.beta * meta_grad

    def log_to_tensorboard(self, iteration, mean_reward, std_reward):
        """
        Logs the metrics to TensorBoard.
        """
        self.writer.add_scalar("Reward/Mean", mean_reward, iteration)
        self.writer.add_scalar("Reward/Std", std_reward, iteration)


# Start timing the computation
start_time = time.time()

# Create task distribution
# scenario_distribution = [Create_scenario(
#     DIST_TYPE) for _ in range(num_scenarios)]
scenario_distribution = [
    {"Dist_Type": "UNIFORM", "min": 8, "max": 11},
    {"Dist_Type": "UNIFORM", "min": 9, "max": 12},
    {"Dist_Type": "UNIFORM", "min": 10, "max": 13},
    {"Dist_Type": "UNIFORM", "min": 11, "max": 14},
    {"Dist_Type": "UNIFORM", "min": 12, "max": 15},
]

test_scenario = {"Dist_Type": "UNIFORM", "min": 9, "max": 14}


# Create environment
env = GymInterface()

# Training the Meta-Learner
meta_learner = MetaLearner(env)

for iteration in range(num_outer_updates):
    # Sample a batch of scenarios
    if len(scenario_distribution) > scenario_batch_size:
        scenario_batch = np.random.choice(
            scenario_distribution, scenario_batch_size, replace=False)
    else:
        scenario_batch = scenario_distribution

    # Adapt the meta-policy to each scenario in the batch
    scenario_models = []
    for scenario in scenario_batch:
        print("\n\nTRAINING SCENARIO: ", scenario)
        print("\nOuter Loop: ", env.cur_outer_loop,
              " / Inner Loop: ", env.cur_inner_loop)
        adapted_model = meta_learner.adapt(scenario)
        scenario_models.append(adapted_model)
        env.cur_episode = 1
        env.cur_inner_loop += 1

    # Perform the meta-update step
    meta_learner.meta_update(scenario_models)

    # Print progress and log to TensorBoard
    # eval_scenario = Create_scenario(DIST_TYPE)

    # Set the scenario for the environment
    meta_learner.env.scenario = test_scenario
    print("\n\nTEST SCENARIO: ", meta_learner.env.scenario)
    env.cur_episode = 1
    env.cur_inner_loop = 1
    mean_reward, std_reward = gw.evaluate_model(
        meta_learner.meta_model, meta_learner.env, N_EVAL_EPISODES)
    meta_learner.logger.record("iteration", iteration)
    meta_learner.logger.record("mean_reward", mean_reward)
    meta_learner.logger.record("std_reward", std_reward)
    meta_learner.logger.dump()
    meta_learner.log_to_tensorboard(iteration, mean_reward, std_reward)
    print(
        f'Iteration {iteration+1}/{num_outer_updates} - Mean Reward: {mean_reward:.2f} ± {std_reward:.2f}\n')
    env.cur_episode = 1
    env.cur_inner_loop = 1
    env.cur_outer_loop += 1

training_end_time = time.time()
# Save the trained meta-policy
meta_learner.meta_model.save("maml_ppo_model")

print("\nMETA TRAINING COMPLETE \n\n\n")

# Evaluate the trained meta-policy
# eval_scenario = Create_scenario(DIST_TYPE)
# Set the scenario for the environment
meta_learner.env.scenario = test_scenario
mean_reward, std_reward = gw.evaluate_model(
    meta_learner.meta_model, meta_learner.env, N_EVAL_EPISODES)
print(
    f"Mean reward over {N_EVAL_EPISODES} episodes: {mean_reward:.2f} +/- {std_reward:.2f}")

# Calculate computation time and print it
end_time = time.time()

# Log final evaluation results to TensorBoard
meta_learner.logger.record("final_mean_reward", mean_reward)
meta_learner.logger.record("final_std_reward", std_reward)
meta_learner.logger.dump()
meta_learner.log_to_tensorboard(num_outer_updates, mean_reward, std_reward)

print(f"Computation time: {(end_time - start_time)/60:.2f} minutes \n",
      f"Training time: {(training_end_time - start_time)/60:.2f} minutes \n ",
      f"Test time:{(end_time - training_end_time)/60:.2f} minutes")

# Optionally render the environment
env.render()
