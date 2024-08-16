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

# Dictionary to store the log data for each experiment
CSV_LOG = {
    'Experiment Case': [],  # To store the name of the experiment case
    'Mean': [],  # To store the mean of the rewards
    'Variance': [],  # To store the variance of the rewards
    'Holding cost': [],  # To store the percentage of holding costs
    'Process cost': [],  # To store the percentage of process costs
    'Delivery cost': [],  # To store the percentage of delivery costs
    'Order cost': [],  # To store the percentage of order costs
    'Shortage cost': []  # To store the percentage of shortage costs
}


def build_model():
    # Function to build or load the RL model based on the selected algorithm
    if RL_ALGORITHM == "PPO":
        # Load the model from the saved path
        '''
        model = PPO.load(os.path.join(
            SAVED_MODEL_PATH, LOAD_MODEL_NAME), env=env)
        print(f"{LOAD_MODEL_NAME} is loaded successfully")
        print(env.observation_space)
        '''

        model = PPO("MlpPolicy", env, verbose=0, n_steps=SIM_TIME *
                    4, learning_rate=0.0001, batch_size=20)

    elif RL_ALGORITHM == "DQN":
        pass  # Placeholder for DQN algorithm implementation

    elif RL_ALGORITHM == "DDPG":
        pass  # Placeholder for DDPG algorithm implementation

    return model


def write_cost_dict():
    # Function to calculate and log the average costs for each cost type
    cost_avg = {
        'Holding cost': 0,
        'Process cost': 0,
        'Delivery cost': 0,
        'Order cost': 0,
        'Shortage cost': 0
    }
    total_cost = 0

    # Calculate average costs over all evaluation episodes
    for x in range(N_EVAL_EPISODES):
        for key in COST_HISTORY[x].keys():
            cost_avg[key] += COST_HISTORY[x][key]
        total_cost += sum(COST_HISTORY[x].values())

    # Calculate the percentage and average for each cost type
    for key in cost_avg.keys():
        CSV_LOG[key].append((cost_avg[key] / total_cost) * 100)
        cost_avg[key] = cost_avg[key] / N_EVAL_EPISODES


# Initialize the environment
env = GymInterface()

# List of order quantities to test
q = [1, 2, 3, 4]

# Load or build the model
model = build_model()

'''
# Simulate the model's performance
mean, std_reward = gw.evaluate_model(model, env, N_EVAL_EPISODES)

# Log the results of the simulation
CSV_LOG['Experiment Case'].append('PPO')
CSV_LOG['Mean'].append(mean)
CSV_LOG['Variance'].append(std_reward)

# Log the cost data
write_cost_dict()
'''

# Iterate over different order quantities and evaluate the model
for action in q:
    COST_HISTORY.clear()
    ORDER_QTY.clear()
    for x in range(MAT_COUNT):
        ORDER_QTY.append(action)

    mean, std_reward = gw.evaluate_model(model, env, N_EVAL_EPISODES)
    CSV_LOG['Experiment Case'].append(f'Q{action}')
    CSV_LOG['Mean'].append(mean)
    CSV_LOG['Variance'].append(std_reward)
    write_cost_dict()

# Save the log data to a CSV file
df = pd.DataFrame(CSV_LOG)
df.to_csv("./Experiment_Result.csv")
