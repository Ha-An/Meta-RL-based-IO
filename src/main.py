import GymWrapper as gw
import time
import HyperparamTuning as ht  # Module for hyperparameter tuning
from GymWrapper import GymInterface
from config_SimPy import *
from config_RL import *
from stable_baselines3 import DQN, DDPG, PPO
from log_SimPy import *
from log_RL import *

# Function to build the model based on the specified reinforcement learning algorithm


def build_model():
    if RL_ALGORITHM == "DQN":
        model = DQN("MlpPolicy", env, verbose=0)
        # model = DQN("MlpPolicy", env, learning_rate=BEST_PARAMS['learning_rate'], gamma=BEST_PARAMS['gamma'],
        #             batch_size=BEST_PARAMS['batch_size'], verbose=0)
    elif RL_ALGORITHM == "DDPG":
        model = DQN("MlpPolicy", env, verbose=0,)
        # model = DDPG("MlpPolicy", env, learning_rate=BEST_PARAMS['learning_rate'], gamma=BEST_PARAMS['gamma'],
        #              batch_size=BEST_PARAMS['batch_size'], verbose=0)
    elif RL_ALGORITHM == "PPO":
        # [Train 1] # model = PPO("MlpPolicy", env, verbose=0, n_steps=SIM_TIME) DEFAULT: learning_rate=0.0003, batch_size=64 => 28 mins
        # [Train 2] # model = PPO("MlpPolicy", env, verbose=0, n_steps=SIM_TIME, learning_rate=0.0001, batch_size=16) => 50 mins
        # [Train 3] # model = PPO("MlpPolicy", env, verbose=0, n_steps=SIM_TIME, learning_rate=0.0002, batch_size=16) => 49 mins
        # [Train 4] # model = PPO("MlpPolicy", env, verbose=0, n_steps=SIM_TIME, learning_rate=0.00015, batch_size=20) => 44 mins
        # [Train 5] # model = PPO("MlpPolicy", env, verbose=0, n_steps=SIM_TIME, learning_rate=0.0001, batch_size=20) => 39 mins
        # [Train 6] # => 40 mins
        model = PPO("MlpPolicy", env, verbose=0, n_steps=SIM_TIME *
                    4, learning_rate=0.0001, batch_size=20)
        # [Train 7] # model = PPO("MlpPolicy", env, verbose=0, n_steps=SIM_TIME*2, learning_rate = 0.0001, batch_size = 20) => 36 mins
        # [Train 8] # model = PPO("MlpPolicy", env, verbose=0, n_steps=SIM_TIME*10, learning_rate = 0.0001, batch_size = 20) => 40 mins

        # model = PPO("MlpPolicy", env, learning_rate=BEST_PARAMS['learning_rate'], gamma=BEST_PARAMS['gamma'],
        #             batch_size=BEST_PARAMS['batch_size'], n_steps=SIM_TIME, verbose=0)

    return model


'''
def export_report(inventoryList):
    for x in range(len(inventoryList)):
        for report in DAILY_REPORTS:
            export_Daily_Report.append(report[x])
    daily_reports = pd.DataFrame(export_Daily_Report)
    daily_reports.columns = ["Day", "Name", "Type",
                         "Start", "Income", "Outcome", "End"]
    daily_reports.to_csv("./Daily_Report.csv")
'''


# Start timing the computation
start_time = time.time()

# Create environment
env = GymInterface()

# Run hyperparameter optimization if enabled
if OPTIMIZE_HYPERPARAMETERS:
    ht.run_optuna()
    # Calculate computation time and print it
    end_time = time.time()
    print(f"Computation time: {(end_time - start_time)/60:.2f} minutes ")
else:
    # Build the model
    if LOAD_MODEL:
        if RL_ALGORITHM == "DQN":
            model = DQN.load(os.path.join(
                SAVED_MODEL_PATH, LOAD_MODEL_NAME), env=env)

        elif RL_ALGORITHM == "DDPG":
            model = DDPG.load(os.path.join(
                SAVED_MODEL_PATH, LOAD_MODEL_NAME), env=env)

        elif RL_ALGORITHM == "PPO":
            model = PPO.load(os.path.join(
                SAVED_MODEL_PATH, LOAD_MODEL_NAME), env=env)
        print(f"{LOAD_MODEL_NAME} is loaded successfully")
    else:
        print("CORRECTION: ", USE_CORRECTION)
        print("Env_Reset: ", env.reset())
        model = build_model()
        # Train the model
        model.learn(total_timesteps=SIM_TIME * N_EPISODES,)
        if SAVE_MODEL:
            model.save(os.path.join(SAVED_MODEL_PATH, SAVED_MODEL_NAME))
            print(f"{SAVED_MODEL_NAME} is saved successfully")

        if STATE_TRAIN_EXPORT:
            gw.export_state('TRAIN')
    training_end_time = time.time()

    # Evaluate the trained model
    mean_reward, std_reward = gw.evaluate_model(model, env, N_EVAL_EPISODES)
    print(
        f"Mean reward over {N_EVAL_EPISODES} episodes: {mean_reward:.2f} +/- {std_reward:.2f}")
    # Calculate computation time and print it
    end_time = time.time()
    print(f"Computation time: {(end_time - start_time)/60:.2f} minutes \n",
          f"Training time: {(training_end_time - start_time)/60:.2f} minutes \n ",
          f"Test time:{(end_time - training_end_time)/60:.2f} minutes")

# Optionally render the environment
env.render()
