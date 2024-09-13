import GymWrapper as gw
import time
import HyperparamTuning as ht  # Module for hyperparameter tuning
from GymWrapper import GymInterface
from config_SimPy import *
from config_RL import *
from stable_baselines3 import DQN, DDPG, PPO
from log_SimPy import *
from log_RL import *
import pandas as pd
from torch.utils.tensorboard import SummaryWriter
# Function to build the model based on the specified reinforcement learning algorithm


def build_model(env):
    if RL_ALGORITHM == "PPO":
        model = PPO("MlpPolicy", env, verbose=0, n_steps=SIM_TIME *
                    4, learning_rate=0.0001, batch_size=20)  # DEFAULT
        print(env.observation_space)
    elif RL_ALGORITHM == "DQN":
        pass
    elif RL_ALGORITHM == "DDPG":
        pass
    return model


def experiment(model):

    # Start timing the computation
    start_time = time.time()
    model.learn(total_timesteps=SIM_TIME * test_steps)
    training_end_time = time.time()

    # Evaluate the trained model
    mean_reward, std_reward = gw.evaluate_model(
        adaptated_model, env, N_EVAL_EPISODES)
    test_end_time = time.time()

    print(
        f"Mean reward over {N_EVAL_EPISODES} episodes: {mean_reward:.2f} +/- {std_reward:.2f}")

    # Calculate computation time and print it
    print(f"Computation time: {(test_end_time - start_time)/60:.2f} minutes \n",
          f"Training time: {(training_end_time - start_time)/60:.2f} minutes \n ",
          f"Test time:{(test_end_time - training_end_time)/60:.2f} minutes")

    return mean_reward, std_reward


# ===================================================================
# Learning Steps for Experiment
test_steps = 5000   # N_EPISODES
scenario = 'AP3'
experiment_result = {'MAML_MEAN_REWARD': [],
                     'MAML_STD_REWARD': [],
                     'DRL_MEAN_REWARD': [],
                     'DRL_STD_REWARD': []
                     }


print("==============MAML_EXPERIMENT==============")
# create environment
env = GymInterface()

# Maml Tensorboard Path
path = os.path.join(TENSORFLOW_LOGS, f'{scenario}_MAML')
path = save_path(path)
env.writer = SummaryWriter(log_dir=path)

# Build Model
adaptated_model = build_model(env)

# Load Model
parent_dir = os.path.dirname(current_dir)
# model_path = os.path.join(parent_dir, 'Saved_Model/MAML_PPO_AP1_E10_O1000')
model_path = os.path.join(
    parent_dir, 'Tensorboard_logs_Experiment_MAML/Train_9/AP3_S7/AP3_S7_O1000')

# Load Adapted model
adaptated_saved_model = PPO.load(model_path)
adaptated_model.policy.load_state_dict(
    adaptated_saved_model.policy.state_dict())

# ADAPTATED Model Test
mean_reward, std_reward = experiment(adaptated_model)

# Save Result
experiment_result['MAML_MEAN_REWARD'].append(mean_reward)
experiment_result['MAML_STD_REWARD'].append(std_reward)


print("==============DRL_EXPERIMENT==============")
# create environment
env = GymInterface()

# DRL Tensorboard Path
path = os.path.join(TENSORFLOW_LOGS, f'{scenario}_DRL')
path = save_path(path)
env.writer = SummaryWriter(log_dir=path)

# Build Model
DRL_model = build_model(env)

# DRL Model Test
mean_reward, std_reward = experiment(DRL_model)

# Save Result
experiment_result['DRL_MEAN_REWARD'].append(mean_reward)
experiment_result['DRL_STD_REWARD'].append(std_reward)


# Save Data Frame
df = pd.DataFrame(experiment_result)
df.index = [scenario]
df.to_csv(os.path.join(TENSORFLOW_LOGS, f"experiment_result_{scenario}.csv"))
