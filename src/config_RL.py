import os
import shutil
from config_SimPy import *

# Using correction option
USE_CORRECTION = False


def Create_scenario():
    if DEMAND_DIST_TYPE == "UNIFORM":
        # Uniform distribution
        # param_min = random.randint(9, 14)
        # param_max = random.randint(param_min, 14)
        param_min = random.randint(10, 13)
        param_max = random.randint(param_min, 13)
        demand_dist = {"Dist_Type": DEMAND_DIST_TYPE,
                       "min": param_min, "max": param_max}
    elif DEMAND_DIST_TYPE == "GAUSSIAN":
        # Gaussian distribution
        param_mean = random.randint(9, 13)
        param_std = random.randint(0, 5)
        demand_dist = {"Dist_Type": DEMAND_DIST_TYPE,
                       "mean": param_mean, "std": param_std}

    if LEAD_DIST_TYPE == "UNIFORM":
        # Uniform distribution
        param_min = random.randint(1, 3)
        param_max = random.randint(param_min, 3)
        leadtime_dist = {"Dist_Type": LEAD_DIST_TYPE,
                         "min": param_min, "max": param_max}
    elif LEAD_DIST_TYPE == "GAUSSIAN":
        # Gaussian distribution
        # Lead time의 최대 값은 Action Space의 최대 값과 곱하였을 때 INVEN_LEVEL_MAX의 2배를 넘지 못하게 설정 해야 함 (INTRANSIT이 OVER되는 현상을 방지 하기 위해서)
        param_mean = random.randint(2, 6)
        param_std = random.randint(0, 3)
        leadtime_dist = {"Dist_Type": LEAD_DIST_TYPE,
                         "mean": param_mean, "std": param_std}
    scenario = {"DEMAND": demand_dist, "LEADTIME": leadtime_dist}

    return scenario


def DEFINE_FOLDER(folder_name):
    if os.path.exists(folder_name):
        file_list = os.listdir(folder_name)
        folder_name = os.path.join(folder_name, f"Train_{len(file_list)+1}")
    else:
        folder_name = os.path.join(folder_name, "Train_1")
    return folder_name


def save_path(path):
    if os.path.exists(path):
        shutil.rmtree(path)
    # Create a new folder
    os.makedirs(path)
    return path


# Episode
N_EPISODES = 5000  # 3000

# RL algorithms
RL_ALGORITHM = "PPO"  # "DP", "DQN", "DDPG", "PPO", "SAC"
# Assembly Process 3
# BEST_PARAMS = {'LEARNING_RATE': 0.0006695881981942652,
#                'GAMMA': 0.917834573740, 'BATCH_SIZE': 8, 'N_STEPS': 600}

# Lead time의 최대 값은 Action Space의 최대 값과 곱하였을 때 INVEN_LEVEL_MAX의 2배를 넘지 못하게 설정 해야 함 (INTRANSIT이 OVER되는 현상을 방지 하기 위해서)
ACTION_SPACE = [0, 1, 2, 3, 4, 5]

DRL_TENSORBOARD = True

# Hyperparameter optimization
OPTIMIZE_HYPERPARAMETERS = False
N_TRIALS = 20  # 50

# Evaluation
N_EVAL_EPISODES = 10  # 100

# Export files
DAILY_REPORT_EXPORT = False
STATE_TRAIN_EXPORT = False
STATE_TEST_EXPORT = False

# Define parent dir's path
current_dir = os.path.dirname(__file__)
parent_dir = os.path.dirname(current_dir)

# Define dir's path
if DRL_TENSORBOARD == True:
    tensorboard_folder = os.path.join(
        parent_dir, "NEW_DRL_tensorboard_logs")
elif DRL_TENSORBOARD == False:
    tensorboard_folder = os.path.join(
        parent_dir, "NEW_META_tensorboard_logs")

result_csv_folder = os.path.join(parent_dir, "result_CSV")
STATE_folder = os.path.join(result_csv_folder, "state")
daily_report_folder = os.path.join(result_csv_folder, "daily_report")

# Define dir's path
TENSORFLOW_LOGS = DEFINE_FOLDER(tensorboard_folder)

STATE = save_path(STATE_folder)
REPORT_LOGS = save_path(daily_report_folder)


# Visualize_Graph
VIZ_INVEN_LINE = True
VIZ_INVEN_PIE = True
VIZ_COST_PIE = True
VIZ_COST_BOX = True

# Saved Model
SAVED_MODEL_PATH = os.path.join(parent_dir, "Saved_Model")
SAVE_MODEL = False
# SAVED_MODEL_NAME = "PPO_Default_1000"
SAVED_MODEL_NAME = "MAML_PPO_AP3_E5_O1000"

# Load Model
LOAD_MODEL = False
LOAD_MODEL_NAME = "E1_MAML_PPO"

# Non-stationary demand
mean_demand = 100
standard_deviation_demand = 20


# tensorboard --logdir="~\tensorboard_log"
# http://localhost:6006/
