import random
import GymWrapper as gw
import time
#import HyperparamTuning as ht  # Module for hyperparameter tuning
from GymWrapper import GymInterface
from config_SimPy import *
from config_RL import *
from stable_baselines3 import DQN, DDPG, PPO
from log_SimPy import *
from log_RL import *
import torch
import learn2learn as l2l
from torch.utils.tensorboard import SummaryWriter


def Make_Task():
    task=[]
    for mean in range(7,15):
        for high in range(0,5):
            for low in range(0,5):
               Dist_info={"Dist_Type":0,
               "Mean":mean,
               "High":high,
               'Low':low}
               task.append(Dist_info)

    for mean in range(7,15):
        for sigma in range(0,5):
            Dist_info={"Dist_Type":1,
             "Mean":mean,
             "Sigma":sigma
            }
            task.append(Dist_info)
    return task

Dist_info=Make_Task()

# Create environment
env = GymInterface()

# Define a function to create new instances of the model
def create_model():
    return PPO('MlpPolicy', env, verbose=0,n_steps=SIM_TIME)

def train_task(model, env, Dist_info, steps=1):

    optimizer = model.policy.optimizer  # Get the optimizer

    for _ in range(steps):
        env.Dist_info = Dist_info
        obs = env.reset()
        done = False
        rewards = []

        while not done:
            action, _ = model.predict(obs)
            obs, reward, done, _ = env.step(action)
            rewards.append(reward ** 2)

        optimizer.zero_grad()
        reward_tensor = torch.tensor(rewards, dtype=torch.float64)
        loss = torch.mean(reward_tensor)  # Maximizing cumulative rewards
        loss.backward()
        optimizer.step()

    return model

# Meta-learning training loop with adaptation and training phases
def meta_train(maml, env, Dist_info,epochs=1000):
    opt = torch.optim.Adam(maml.module.policy.state_dict(), lr=0.001)
    writer = SummaryWriter(log_dir=TENSORFLOW_LOGS)
    for epoch in range(epochs):
        meta_train_loss = 0.0
        for x in range(len(Dist_info)):
            learner = maml.clone()
            env.Dist_info=Dist_info[x]
            obs = env.reset()
            done = False
            episode_reward = 0
            learner = train_task(learner, env, Dist_info[x], steps=1)
            while not done:
                action, _ = learner.predict(obs)
                obs, reward, done, _ = env.step(action)
                episode_reward += reward

            opt.zero_grad()
            loss = -episode_reward
            loss.backward()
            opt.step()

            meta_train_loss += loss.item()
            writer.add_scalar(
                "loss", loss, global_step=epoch)
            # Log each cost ratio at the end of the episode
        print(f"Epoch {epoch + 1}/{epochs}, Meta-Train Loss: {meta_train_loss / len(Dist_info)}")



# Wrap the model with MAML
model=create_model()
maml = l2l.algorithms.MAML(model, lr=0.001, first_order=False)
print(maml)
# Start meta-training
meta_train(maml, env,Dist_info)
