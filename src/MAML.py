import GymWrapper as gw
import time
from GymWrapper import GymInterface
from config_SimPy import *
from config_RL import *
from log_SimPy import *
from log_RL import *
from Def_Scenarios import *
import numpy as np
import torch
import torch.nn.functional as F
from inner_Model import *
from gym import spaces
from inner_Model import *
from meta_model import *
from torch.utils.tensorboard import SummaryWriter


K = N_EPISODES
# Hyperparameters
ALPHA = 0.0001  # Inner loop step size (사용되지 않는 값) ->  SB3 PPO 기본 값(0.0003)
BATCH_SIZE = 20  # Default 64
N_STEPS = SIM_TIME*K  # Default 2048

BETA = 0.0001  # Outer loop step size ## Default: 0.001
train_scenario_batch_size = 10  # Batch size for random chosen scenarios
test_scenario_batch_size = 5  # Batch size for random chosen scenarios
num_outer_updates = 1000  # Number of outer loop updates -> meta-training iterations
tensor_save_path = TENSORFLOW_LOGS
model_name = SAVED_MODEL_NAME
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
        self.action_dims = [len(ACTION_SPACE) for _ in range(MAT_COUNT)]  # MultiDiscrete
        self.state_dim = len(self.env.reset())

        self.meta_model = outer_loop(
            state_dim = self.state_dim,
            action_dims = self.action_dims,
            lr = 3e-4,
            gamma = 0.99,
            clip_epsilon = CLIP_EPSILON,
            update_steps = 10
        )
        self.writer = SummaryWriter(log_dir=tensor_save_path)

    def inner_loop(self, k = K):
        self.env.reset()
        model = inner_model(self.env, n_steps = 10, lr = self.alpha, batch_size = BATCH_SIZE, total_steps = 100)
        model.main_module()
        return model

    def meta_update(self, adapted_model):
        self.meta_model.update_steps = 10      
        self.meta_model.device = DEVICE

        self.meta_model.update(adapted_model)

    def meta_test(self, iteration, test_scenario_batch):
        """
        Performs the meta-test step by averaging gradients across scenarios.
        """
        # eval_scenario = Create_scenario(DIST_TYPE)
        # test_scenario_batch = [Create_scenario()
        #                        for _ in range(test_scenario_batch_size)]

        # Set the scenario for the environment
        all_rewards = []
        for test_scenario in test_scenario_batch:
            # for test_scenario in test_scenario_batch:
            state = self.env.reset()
            self.env.scenario = test_scenario
            episode_reward = 0
            print("\n\nTEST SCENARIO: ", self.env.scenario)
            for x in range(SIM_TIME):
                action = self.meta_model.select_action(state)
                state, reward, done, _ = self.env.step(action)
                episode_reward += reward
            all_rewards.append(episode_reward)

        # Calculate mean reward across all episodes
        meta_mean_reward = np.mean(all_rewards)
        std_reward = np.std(all_rewards)
        self.log_to_tensorboard(iteration, meta_mean_reward, std_reward)

        return meta_mean_reward, std_reward

    def log_to_tensorboard(self, iteration, mean_reward, std_reward):
        """
        Logs the metrics to TensorBoard.
        """
        self.writer.add_scalar("Reward/Mean", mean_reward, iteration)
        self.writer.add_scalar("Reward/Std", std_reward, iteration)


def main():
    # Start timing the computation
    start_time = time.time()
    # Create environment
    env = GymInterface()
    # Training the Meta-Learner
    meta_learner = MetaLearner(env)
    meta_rewards = []
    random_rewards = []
    # Generate scenarios
    all_scenarios = create_scenarios()
    print(f"Total {len(all_scenarios)} scenarios have been generated.")
    # Split scenarios into 8:2 ratio
    train_scenarios, test_scenarios = split_scenarios(all_scenarios)
    print(f"Number of training scenarios: {len(train_scenarios)}")
    print(f"Number of test scenarios: {len(test_scenarios)}")
    for iteration in range(num_outer_updates):
        env.scenario_batch_size = train_scenario_batch_size
        # LINE 3: Sample a batch of scenarios
        train_scenario_batch = random.sample(
            train_scenarios, train_scenario_batch_size)
        # scenario_batch = [Create_scenario()
        #                   for _ in range(train_scenario_batch_size)]
        if iteration == num_outer_updates-1:
            meta_learner.env.outer_end = True
        # Adapt the meta-policy to each scenario in the batch

        for scenario in train_scenario_batch:  # LINE 4
            print("\n\nTRAINING SCENARIO(DEMAND): ", scenario["DEMAND"])
            print("\n\nTRAINING SCENARIO(LEAD_TIME): ", scenario["LEADTIME"])
            print("\nOuter Loop: ", env.cur_outer_loop,
                  " / Inner Loop: ", env.cur_inner_loop)
            # Reset the scenario for the environment
            meta_learner.env.scenario = scenario
            print("Demand: ", meta_learner.env.scenario["DEMAND"])
            print("Lead_time: ", meta_learner.env.scenario["LEADTIME"])
            # LINE 5 - 7
            adapted_model = meta_learner.inner_loop()  # LINE 6-7
            # LINE 8: 학습된 모델로부터 rollout 수집
            # rollout buffer에는 K개의 에피소드가 저장되어 있음
            meta_learner.meta_update(adapted_model)
            # rollout_list.append(rollout_buffer)
            '''
            # LINE 8: 학습된 모델로부터 rollout 수집
            rollout_buffer = adapted_model.rollout_buffer
            rollout_list.append(rollout_buffer)
            '''
            env.cur_episode = 1
            env.cur_inner_loop += 1
        # # LINE 10: Perform the meta-update step
        # meta_learner.meta_update(rollout_list)

        # Evaluate the meta-policy on the test scenario
        test_scenario_batch = random.sample(
            test_scenarios, test_scenario_batch_size)
        meta_mean_reward, meta_std_reward = meta_learner.meta_test(
            iteration, test_scenario_batch)
        meta_rewards.append(meta_mean_reward)
        print(
            f'Iteration {iteration+1}/{num_outer_updates} - Mean Reward: {meta_mean_reward:.2f} ± {meta_std_reward:.2f}\n')
        print('===========================================================')
        env.cur_episode = 1
        env.cur_inner_loop = 1
        env.cur_outer_loop += 1
        env.outer_end = False
        # Save the trained meta-policy
        if EXPERIMENT_MAML == False:
            torch.save(meta_learner.meta_model.policy.state_dict(), model_name)

        else:
            torch.save(meta_learner.meta_model.policy.state_dict(), os.path.join(tensor_save_path, model_name))
    training_end_time = time.time()
    print("\nMETA TRAINING COMPLETE \n\n\n")
    # Calculate computation time and print it
    end_time = time.time()

    # Evaluate the trained meta-policy
    meta_mean_reward, meta_std_reward = meta_learner.meta_test(
        num_outer_updates, test_scenario_batch)
    print(
        f"Mean reward over {N_EVAL_EPISODES} episodes: {meta_mean_reward:.2f} +/- {meta_std_reward:.2f}")

    print(f"Computation time: {(end_time - start_time)/60:.2f} minutes \n",
          f"Training time: {(training_end_time - start_time)/60:.2f} minutes \n ",
          f"Test time:{(end_time - training_end_time)/60:.2f} minutes")
    # Optionally render the environment
    env.render()

    return meta_mean_reward, meta_std_reward

'''
if EXPERIMENT_MAML == False:
    main()  # default
else:
    pass
'''
main()