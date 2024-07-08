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
import torch.nn.functional as F
from stable_baselines3 import PPO
from stable_baselines3.common.logger import configure
from stable_baselines3.common.buffers import RolloutBuffer
from stable_baselines3.common.buffers import BaseBuffer
from stable_baselines3.common.type_aliases import RolloutBufferSamples
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3.common.callbacks import BaseCallback
from gym import spaces
from stable_baselines3.common.utils import explained_variance
from stable_baselines3.common.preprocessing import get_action_dim, get_obs_shape

from torch.utils.tensorboard import SummaryWriter

# Hyperparameters
ALPHA = 0.002  # Inner loop step size (사용되지 않는 값) ->  SB3 PPO 기본 값(0.0003)
BATCH_SIZE = 128  # Default 64

BETA = 0.001  # Outer loop step size ## Default: 0.001
num_scenarios = 11  # Number of full scenarios for meta-training
scenario_batch_size = 4  # Batch size for random chosen scenarios
num_inner_updates = N_EPISODES  # Number of gradient steps for adaptation
num_outer_updates = 400  # Number of outer loop updates -> meta-training iterations


class LastRolloutInfoCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(LastRolloutInfoCallback, self).__init__(verbose)
        self.last_rollout_info = None

    def _on_step(self) -> bool:
        return True  # 계속 학습을 진행합니다.

    def _on_rollout_end(self) -> None:
        buffer = self.model.rollout_buffer
        self.last_rollout_info = {
            "observations": buffer.observations[-1].copy(),
            "actions": buffer.actions[-1].copy(),
            "rewards": buffer.rewards[-1].copy(),
            "advantages": buffer.advantages[-1].copy(),
            "returns": buffer.returns[-1].copy(),
            "episode_starts": buffer.episode_starts[-1].copy(),
            "values": buffer.values[-1].copy(),
            "log_probs": buffer.log_probs[-1].copy()
        }

    def get_last_rollout_info(self):
        if self.last_rollout_info is None:
            return "No rollout information available yet."

        info = "Last Rollout Information:\n"
        for key, value in self.last_rollout_info.items():
            info += f"{key}:\n"
            info += f"  Shape: {value.shape}\n"
            info += f"  Data type: {value.dtype}\n"
            if np.issubdtype(value.dtype, np.number):
                info += f"  Mean: {np.mean(value):.4f}, Std: {np.std(value):.4f}\n"
                info += f"  Min: {np.min(value):.4f}, Max: {np.max(value):.4f}\n"
            if value.size > 0:
                info += f"  Values: {value.flatten()[:5]}\n"
            info += "\n"

        return info

# Meta-learning algorithm


class MetaLearner:
    def __init__(self, env, policy='MlpPolicy', alpha=ALPHA, beta=BETA):
        """
        Initializes the MetaLearner with the specified environment and hyperparameters.
        """
        self.env = env
        self.policy = policy
        self.alpha = alpha
        self.beta = beta

        # CUDA 사용 설정
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        # self.meta_model = PPO(policy, self.env, verbose=0,
        #                       n_steps=SIM_TIME, learning_rate=self.beta, batch_size=BATCH_SIZE, device=self.device)
        self.meta_model = PPO(policy, self.env, verbose=0,
                              n_steps=SIM_TIME, learning_rate=self.beta, batch_size=2, device=self.device)
        self.meta_model._logger = configure(None, ["stdout"])

        self.logger = configure()
        self.writer = SummaryWriter(log_dir='./tensorboard_logs')

    def adapt(self, num_updates=num_inner_updates):
        """
        Adapts the meta-policy to a specific task using gradient descent.
        """
        self.env.reset()
        # self.env.scenario = scenario  # Reset the scenario for the environment
        adapted_model = PPO(self.policy, self.env, verbose=0,
                            n_steps=SIM_TIME, learning_rate=self.alpha, batch_size=2, device=self.device)

        # (1) 전체 모델의 파라미터(정책 네트워크와 가치 함수 네트워크)를 복사
        adapted_model.set_parameters(self.meta_model.get_parameters())
        # (2) 정책 네트워크의 파라미터만 복사
        # adapted_model.policy.load_state_dict(self.meta_model.policy.state_dict())

        # self.verify_models_equality(self.meta_model, adapted_model)

        # 콜백을 사용하여 마지막 rollout 정보를 추적
        # callback = LastRolloutInfoCallback()
        for _ in range(num_updates):
            # Train the policy on the specific scenario
            # adapted_model.learn(total_timesteps=SIM_TIME, callback=callback)
            adapted_model.learn(total_timesteps=SIM_TIME)
        # print(callback.get_last_rollout_info())

        return adapted_model

    def verify_models_equality(self, meta_model, adapted_model):
        # 모델의 상태 딕셔너리를 가져옵니다.
        meta_state_dict = meta_model.policy.state_dict()
        adapted_state_dict = adapted_model.policy.state_dict()

        # 두 상태 딕셔너리의 키가 동일한지 확인합니다.
        if meta_state_dict.keys() != adapted_state_dict.keys():
            print("The models have different structures.")
            return False

        # 각 매개변수를 비교합니다.
        for key in meta_state_dict.keys():
            if not torch.equal(meta_state_dict[key], adapted_state_dict[key]):
                print(f"Mismatch in parameter: {key}")
                return False

        print("All parameters were copied correctly!")
        return True

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

    '''
    def meta_update(self, scenario_models):
        print("Outter_Loop_Start")
        # Learning 100 times
        for outter_itters in range(1):
            # For every scenarios
            # for x in range(len(scenarios)):
            for x in scenario_models:
                obs = self.env.reset()
                self.env.scenario = x["scenario"]
                # collect rollout_buffers
                for _ in range(SIM_TIME):
                    action, _ = x["adapted_model"].predict(
                        obs, deterministic=False)
                    next_obs, reward, done, info = self.env.step(action)
                    obs = next_obs
                    if done:
                        obs = self.env.reset()
                # Enter the rollout buffer of the learned model into meta_model
                self.meta_model.rollout_buffer = x["adapted_model"].rollout_buffer
                # Meta_Model_training
                self.meta_model.train()
        # Save model every outter_Loop Ended
        # meta_learner.meta_model.save("maml_ppo_model")
    '''

    def meta_update(self, rollout_list):

        # 데이터 결합
        combined_observations = np.concatenate(
            [np.array(rollout.observations) for rollout in rollout_list])
        combined_actions = np.concatenate(
            [np.array(rollout.actions) for rollout in rollout_list])
        combined_rewards = np.concatenate(
            [np.array(rollout.rewards) for rollout in rollout_list])
        combined_returns = np.concatenate(
            [np.array(rollout.returns) for rollout in rollout_list])
        combined_values = np.concatenate(
            [np.array(rollout.values) for rollout in rollout_list])
        combined_log_probs = np.concatenate(
            [np.array(rollout.log_probs) for rollout in rollout_list])
        combined_advantages = np.concatenate(
            [np.array(rollout.advantages) for rollout in rollout_list])

        total_steps = len(combined_observations)

        # 데이터를 텐서로 변환
        observations = torch.FloatTensor(
            combined_observations).to(self.meta_model.device)
        actions = torch.FloatTensor(
            combined_actions).to(self.meta_model.device)
        returns = torch.FloatTensor(
            combined_returns).to(self.meta_model.device)
        old_values = torch.FloatTensor(
            combined_values).to(self.meta_model.device)
        old_log_probs = torch.FloatTensor(
            combined_log_probs).to(self.meta_model.device)
        advantages = torch.FloatTensor(
            combined_advantages).to(self.meta_model.device)

        # clip_range 함수 호출
        if callable(self.meta_model.clip_range):
            current_progress_remaining = 1.0  # 또는 적절한 값으로 설정
            clip_range = self.meta_model.clip_range(current_progress_remaining)
        else:
            clip_range = self.meta_model.clip_range

        # PPO 업데이트 단계 수행
        for epoch in range(self.meta_model.n_epochs):
            # 데이터를 섞음
            permutation = torch.randperm(total_steps)

            # 미니배치로 나눔
            for start_idx in range(0, total_steps, self.meta_model.batch_size):
                end_idx = start_idx + self.meta_model.batch_size
                batch_indices = permutation[start_idx:end_idx]

                batch_obs = observations[batch_indices]
                batch_actions = actions[batch_indices]
                batch_returns = returns[batch_indices]
                batch_old_values = old_values[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
                batch_advantages = advantages[batch_indices]

                # 현재 정책에서의 로그 확률 계산
                values, log_prob, entropy = self.meta_model.policy.evaluate_actions(
                    batch_obs, batch_actions)

                # Shape 및 dtype 조정
                batch_returns = batch_returns.float().view(-1)
                values = values.float().view(-1)

                # Advantage 정규화
                batch_advantages = (batch_advantages - batch_advantages.mean()
                                    ) / (batch_advantages.std() + 1e-8)

                # 비율 계산
                ratio = torch.exp(log_prob - batch_old_log_probs)

                # Policy loss 계산
                policy_loss_1 = batch_advantages * ratio
                policy_loss_2 = batch_advantages * \
                    torch.clamp(ratio, 1 - clip_range, 1 + clip_range)
                policy_loss = -torch.min(policy_loss_1, policy_loss_2).mean()

                # Value loss 계산
                value_loss = F.mse_loss(batch_returns, values)

                # Entropy loss 계산
                entropy_loss = -torch.mean(entropy)

                # 총 loss 계산
                loss = policy_loss + self.meta_model.vf_coef * \
                    value_loss + self.meta_model.ent_coef * entropy_loss

                # 그래디언트 계산 및 옵티마이저 단계 수행
                self.meta_model.policy.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.meta_model.policy.parameters(), self.meta_model.max_grad_norm)
                self.meta_model.policy.optimizer.step()

            # Early stopping 조건 체크 (필요하다면)

        # 학습률 감소
        self.meta_model._update_learning_rate(self.meta_model.policy.optimizer)

    def meta_test(self, env):
        """
        Performs the meta-test step by averaging gradients across scenarios.
        """
        # Print progress and log to TensorBoard
        # eval_scenario = Create_scenario(DIST_TYPE)

        self.env.reset()

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


def inspect_rollout_buffers(rollout_list):

    # 데이터 구조 확인
    print("데이터 구조 확인")
    for i, rollout in enumerate(rollout_list):
        print(f"RolloutBuffer {i}:")
        print(f"  observations shape: {rollout.observations.shape}")
        print(f"  actions shape: {rollout.actions.shape}")
        print(f"  rewards shape: {rollout.rewards.shape}")
        print(f"  returns shape: {rollout.returns.shape}")
        print(f"  episode_starts shape: {rollout.episode_starts.shape}")
        print(f"  values shape: {rollout.values.shape}")
        print(f"  log_probs shape: {rollout.log_probs.shape}")
        print(f"  advantages shape: {rollout.advantages.shape}")
        print(f"  pos: {rollout.pos}")
        print()

    # 데이터 타입 확인
    print("데이터 타입 확인")
    for i, rollout in enumerate(rollout_list):
        print(
            f"RolloutBuffer {i} observations dtype: {rollout.observations.dtype}")

    # 데이터 형태 확인
    print("데이터 형태 확인")
    for i, rollout in enumerate(rollout_list):
        print(
            f"RolloutBuffer {i} observations first element shape: {rollout.observations[0].shape}")
        print(f"Sample: {rollout.observations[0]}")


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
    rollout_list = []
    for scenario in scenario_batch:
        print("\n\nTRAINING SCENARIO: ", scenario)
        print("\nOuter Loop: ", env.cur_outer_loop,
              " / Inner Loop: ", env.cur_inner_loop)

        # Reset the scenario for the environment
        meta_learner.env.scenario = scenario
        print("Scenario: ", meta_learner.env.scenario)
        # 특정 시나리오에 대한 학습 진행
        adapted_model = meta_learner.adapt()
        scenario_models.append(adapted_model)

        # 학습된 모델로부터 rollout 수집
        # print("Observation space:", env.observation_space)
        # print("Action space:", env.action_space)

        # rollouts_list.append(rollout_data)
        rollout_buffer = adapted_model.rollout_buffer
        # last_obs = rollout_buffer.observations[-1]
        # last_actions = rollout_buffer.actions[-1]
        # last_rewards = rollout_buffer.rewards[-1]
        # print("last_obs: ", last_obs)
        # print("last_actions: ", last_actions)
        # print("last_rewards: ", last_rewards)
        rollout_list.append(rollout_buffer)

        env.cur_episode = 1
        env.cur_inner_loop += 1

    # Perform the meta-update step
    # inspect_rollout_buffers(rollout_list)
    meta_learner.meta_update(rollout_list)

    # Evaluate the meta-policy on the test scenario
    mean_reward, std_reward = meta_learner.meta_test(env)
    overfitting_diagnosis.append((iteration, mean_reward, std_reward))

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
