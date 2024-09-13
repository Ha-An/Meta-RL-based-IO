import MAML2 as meta
from config_RL import *
from stable_baselines3 import PPO
import os
import pandas as pd

# Experiment case: N_Episodes(inner_loop)
# experiment_case = [5, 10]
# experiment_case = [10]
experiment_case = [3, 7]

experiment_result = {"MEAN": [],
                     'STD': []}

# Scenario number
scenario = 'AP1'
current_dir = os.path.dirname(__file__)
parent_dir = os.path.dirname(current_dir)
origin_log = TENSORFLOW_LOGS

for n_episodes in experiment_case:
    path = os.path.join(origin_log, f'{scenario}_S{n_episodes}')
    path = save_path(path)
    meta.tensor_save_path = path
    meta.N_STEPS = n_episodes * SIM_TIME
    meta.model_name = f'{scenario}_S{n_episodes}_O{meta.num_outer_updates}'
    mean, std = meta.main()

    experiment_result['MEAN'].append(mean)
    experiment_result['STD'].append(std)

df = pd.DataFrame(experiment_result)
df.index = [f'{scenario}_S{x}' for x in range(len(experiment_case))]
df.to_csv(os.path.join(origin_log, f"experiment_result_{scenario}.csv"))
