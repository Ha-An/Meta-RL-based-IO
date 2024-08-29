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
    if RL_ALGORITHM == "PPO":
        model = PPO("MlpPolicy", env, verbose=0, n_steps=SIM_TIME *
                    4, learning_rate=0.0001, batch_size=20)
        print(env.observation_space)
    elif RL_ALGORITHM == "DQN":
        pass
    elif RL_ALGORITHM == "DDPG":
        pass
    return model


# Start timing the computation
start_time = time.time()

# Create environment
env = GymInterface()

# Build the model
model = build_model()

saved_model = PPO.load(os.path.join(
    SAVED_MODEL_PATH, LOAD_MODEL_NAME), env=env)  # Load the saved model
# 정책 네트워크의 파라미터 복사
model.policy.load_state_dict(saved_model.policy.state_dict())

# Train the model
model.learn(total_timesteps=SIM_TIME * N_EPISODES)

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
