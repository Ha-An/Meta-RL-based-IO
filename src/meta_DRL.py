import matplotlib.pyplot as plt
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
from copy import deepcopy
import torch
from torch.optim import Adam
from torch import nn
from stable_baselines3 import PPO
from stable_baselines3.common.logger import configure
from torch.utils.tensorboard import SummaryWriter

# Hyperparameters
alpha = 0.002  # Inner loop step size (사용되지 않는 값) ->  SB3 PPO 기본 값(0.0003)
BATCH_SIZE = 128  # Default 64

beta = 0.001  # Outer loop step size ## Default: 0.001
num_scenarios = 11  # Number of full scenarios for meta-training
scenario_batch_size = 2  # Batch size for random chosen scenarios
num_inner_updates = N_EPISODES  # Number of gradient steps for adaptation
num_outer_updates = 150  # Number of outer loop updates -> meta-training iterations

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
        self.meta_model = PPO(policy, self.env, verbose=0,
                              n_steps=SIM_TIME, learning_rate=self.alpha, batch_size=BATCH_SIZE)
        self.logger = configure()
        self.writer = SummaryWriter(log_dir='./tensorboard_logs')

    def adapt(self, scenario, num_updates=num_inner_updates):
        """
        Adapts the meta-policy to a specific task using gradient descent.
        """
        self.env.scenario = scenario  # Reset the scenario for the environment
        adapted_model = PPO(self.policy, self.env, verbose=0,
                            n_steps=SIM_TIME, learning_rate=self.alpha, batch_size=BATCH_SIZE)

        # 전체 모델의 파라미터(정책 네트워크와 가치 함수 네트워크)를 복사
        adapted_model.set_parameters(self.meta_model.get_parameters())
        # 정책 네트워크의 파라미터만 복사
        # adapted_model.policy.load_state_dict(self.meta_model.policy.state_dict())

        # for _ in range(num_updates):
        #     # Train the policy on the specific scenario
        #     adapted_model.learn(total_timesteps=SIM_TIME)
        adapted_model.learn(total_timesteps=SIM_TIME*num_updates)
        return adapted_model

    # def meta_update(self, scenario_models):
    #     """
    #     Performs the meta-update step by averaging gradients across scenarios.
    #     """
    #     meta_grads = []
    #     for scenario_model in scenario_models:
    #         # Retrieve gradients from the adapted policy
    #         grads = []
    #         for param in scenario_model.policy.parameters():
    #             if param.grad is not None:
    #                 grads.append(param.grad.clone())
    #             else:
    #                 grads.append(torch.zeros_like(param.data))
    #         meta_grads.append(grads)

    #     # Average gradients across tasks
    #     mean_meta_grads = [torch.mean(torch.stack(
    #         meta_grads_i), dim=0) for meta_grads_i in zip(*meta_grads)]

    #     # Update meta-policy parameters using the outer loop learning rate
    #     for param, meta_grad in zip(self.meta_model.policy.parameters(), mean_meta_grads):
    #         param.data -= self.beta * meta_grad

    #     # Zero out the gradients for the next iteration
    #     # self.meta_model.policy.zero_grad()

    def meta_update(self, scenario_models):
        # 각 시나리오 모델에서 얻은 경험을 결합
        combined_rollouts = []
        for scenario_model in scenario_models:
            # 각 시나리오 모델로 환경과 상호작용하여 rollout 수집
            obs = self.env.reset()
            for _ in range(SIM_TIME):
                action, _ = scenario_model.predict(obs, deterministic=False)
                next_obs, reward, done, info = self.env.step(action)
                combined_rollouts.append((obs, action, reward, next_obs, done))
                obs = next_obs
                if done:
                    obs = self.env.reset()

        # 결합된 rollout으로 메타 모델 업데이트
        self.meta_model.learn(total_timesteps=len(
            combined_rollouts), reset_num_timesteps=False)

    '''
    def compute_kl(self, old_policy, new_policy, obs):
        old_dist = old_policy.get_distribution(obs)
        new_dist = new_policy.get_distribution(obs)
        return torch.mean(torch.sum(old_dist.kl_divergence(new_dist), dim=1))

    def meta_update(self, scenario_models):
        old_policy = deepcopy(self.meta_model.policy)

        # 메타 손실 계산
        meta_loss = 0
        for scenario_model in scenario_models:
            obs = self.env.reset()
            done = False
            episode_reward = 0
            while not done:
                action, _ = scenario_model.predict(obs, deterministic=True)
                obs, reward, done, _ = self.env.step(action)
                episode_reward += reward
            meta_loss -= episode_reward
        meta_loss /= len(scenario_models)

        # 그래디언트 계산
        self.meta_model.policy.zero_grad()
        meta_loss.backward()

        # 라인 서치
        step_size = self.beta
        for _ in range(self.backtrack_iters):
            new_policy = deepcopy(old_policy)
            for param, grad in zip(new_policy.parameters(), self.meta_model.policy.parameters()):
                param.data.add_(grad.grad.data, alpha=-step_size)

            # 새로운 메타 손실 계산
            new_meta_loss = 0
            for scenario_model in scenario_models:
                obs = self.env.reset()
                done = False
                episode_reward = 0
                while not done:
                    action, _ = new_policy(obs)
                    obs, reward, done, _ = self.env.step(action)
                    episode_reward += reward
                new_meta_loss -= episode_reward
            new_meta_loss /= len(scenario_models)

            # KL 발산 계산
            obs = self.env.reset()
            kl_div = self.compute_kl(old_policy, new_policy, obs)

            if kl_div <= self.max_kl and new_meta_loss < meta_loss:
                self.meta_model.policy.load_state_dict(new_policy.state_dict())
                print(f"Meta-update accepted with step size {step_size}")
                break

            step_size *= self.backtrack_coeff

        if step_size < self.beta * (self.backtrack_coeff ** (self.backtrack_iters - 1)):
            print("Meta-update rejected")
    '''

    def meta_test(self, env):
        """
        Performs the meta-test step by averaging gradients across scenarios.
        """
        # Print progress and log to TensorBoard
        # eval_scenario = Create_scenario(DIST_TYPE)

        # Set the scenario for the environment
        self.env.scenario = test_scenario
        print("\n\nTEST SCENARIO: ", self.env.scenario)
        env.cur_episode = 1
        env.cur_inner_loop = 1
        mean_reward, std_reward = gw.evaluate_model(
            self.meta_model, self.env, N_EVAL_EPISODES)
        self.logger.record("iteration", iteration)
        self.logger.record("mean_reward", mean_reward)
        self.logger.record("std_reward", std_reward)
        self.logger.dump()
        self.log_to_tensorboard(iteration, mean_reward, std_reward)
        print(
            f'Iteration {iteration+1}/{num_outer_updates} - Mean Reward: {mean_reward:.2f} ± {std_reward:.2f}\n')
        env.cur_episode = 1
        env.cur_inner_loop = 1
        env.cur_outer_loop += 1

        return mean_reward, std_reward

    def log_to_tensorboard(self, iteration, mean_reward, std_reward):
        """
        Logs the metrics to TensorBoard.
        """
        self.writer.add_scalar("Reward/Mean", mean_reward, iteration)
        self.writer.add_scalar("Reward/Std", std_reward, iteration)


# Start timing the computation
start_time = time.time()

# Create task distribution
scenario_distribution = [Create_scenario(
    DIST_TYPE) for _ in range(num_scenarios)]
scenario_distribution = [
    {"Dist_Type": "UNIFORM", "min": 8, "max": 10},
    {"Dist_Type": "UNIFORM", "min": 9, "max": 11},
    {"Dist_Type": "UNIFORM", "min": 10, "max": 12},
    {"Dist_Type": "UNIFORM", "min": 11, "max": 13},
    {"Dist_Type": "UNIFORM", "min": 12, "max": 14},
    {"Dist_Type": "UNIFORM", "min": 13, "max": 15},
    {"Dist_Type": "UNIFORM", "min": 8, "max": 11},
    {"Dist_Type": "UNIFORM", "min": 9, "max": 12},
    {"Dist_Type": "UNIFORM", "min": 10, "max": 13},
    {"Dist_Type": "UNIFORM", "min": 11, "max": 14},
    {"Dist_Type": "UNIFORM", "min": 12, "max": 15}
]
test_scenario = {"Dist_Type": "UNIFORM", "min": 9, "max": 14}
# scenario_distribution = [
#     {"Dist_Type": "UNIFORM", "min": 8, "max": 8},
#     {"Dist_Type": "UNIFORM", "min": 10, "max": 10},
#     {"Dist_Type": "UNIFORM", "min": 11, "max": 11},
#     {"Dist_Type": "UNIFORM", "min": 13, "max": 13},
#     {"Dist_Type": "UNIFORM", "min": 15, "max": 15},
# ]
# test_scenario = {"Dist_Type": "UNIFORM", "min": 12, "max": 12}


# scenario_distribution = [
#     {"Dist_Type": "GAUSSIAN", "mean": 8, "std": 11},
#     {"Dist_Type": "GAUSSIAN", "mean": 9, "std": 12},
#     {"Dist_Type": "GAUSSIAN", "mean": 10, "std": 13},
#     {"Dist_Type": "GAUSSIAN", "mean": 11, "std": 14},
#     {"Dist_Type": "GAUSSIAN", "mean": 12, "std": 15},
# ]
# test_scenario = {"Dist_Type": "GAUSSIAN", "mean": 9, "std": 14}


# Create environment
env = GymInterface()

# Training the Meta-Learner
meta_learner = MetaLearner(env)
overfitting_diagnosis = []

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

    # Evaluate the meta-policy on the test scenario
    mean_reward, std_reward = meta_learner.meta_test(env)
    overfitting_diagnosis.append((iteration, mean_reward, std_reward))

    # # Print progress and log to TensorBoard
    # # eval_scenario = Create_scenario(DIST_TYPE)

    # # Set the scenario for the environment
    # meta_learner.env.scenario = test_scenario
    # print("\n\nTEST SCENARIO: ", meta_learner.env.scenario)
    # env.cur_episode = 1
    # env.cur_inner_loop = 1
    # mean_reward, std_reward = gw.evaluate_model(
    #     meta_learner.meta_model, meta_learner.env, N_EVAL_EPISODES)
    # meta_learner.logger.record("iteration", iteration)
    # meta_learner.logger.record("mean_reward", mean_reward)
    # meta_learner.logger.record("std_reward", std_reward)
    # meta_learner.logger.dump()
    # meta_learner.log_to_tensorboard(iteration, mean_reward, std_reward)
    # print(
    #     f'Iteration {iteration+1}/{num_outer_updates} - Mean Reward: {mean_reward:.2f} ± {std_reward:.2f}\n')
    # env.cur_episode = 1
    # env.cur_inner_loop = 1
    # env.cur_outer_loop += 1

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

# Check for meta overfitting
iterations, mean_rewards, std_rewards = zip(*overfitting_diagnosis)
plt.errorbar(iterations, mean_rewards, yerr=std_rewards,
             fmt='-o', label='Test Scenario Reward')
plt.xlabel('Iteration')
plt.ylabel('Mean Reward')
plt.title('Meta Overfitting Diagnosis')
plt.legend()
plt.show()
