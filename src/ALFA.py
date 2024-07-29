import GymWrapper as gw
import time
from GymWrapper import GymInterface
from config_SimPy import *
from config_RL import *
from log_SimPy import *
from log_RL import *
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from stable_baselines3 import PPO
from stable_baselines3.common import on_policy_algorithm
from gym import spaces
from stable_baselines3.common.callbacks import BaseCallback
from torch.utils.tensorboard import SummaryWriter

class CustomCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(CustomCallback, self).__init__(verbose)

    def _on_rollout_start(self):
        print("Rollout started")

    def _on_rollout_end(self):
        print("Rollout ended")

    def _on_step(self):
        # This method is called at each environment step (i.e. for each action taken)
        return True

# Hyperparameters
BATCH_SIZE = 20  # Default 64
N_STEPS = SIM_TIME*4  # Default 2048

train_num_scenarios = 20  # Number of full scenarios for meta-training
test_num_scenarios = 5  # Number of full scenarios for meta-training
scenario_batch_size = 2  # Batch size for random chosen scenarios
num_inner_updates = N_EPISODES  # Number of gradient steps for adaptation
num_outer_updates = 10  # Number of outer loop updates -> meta-training iterations


class Meta_Model(nn.Module):
    def __init__(self, lr=0.001):
        super(Meta_Model, self).__init__()
        # Build meta model
        self.fc1 = nn.Linear(2, 64)
        self.fc2 = nn.Linear(64, 128)
        self.fc3 = nn.Linear(128, 2)  # Single output unit for ratio prediction
        
        # Use Adam optimizer
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        output = torch.sigmoid(self.fc3(x))  # Sigmoid activation to ensure output is in range [0, 1]
        return output
    
    def update_model(self, loss): # 'loss' is total loss of this task 
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

class Base_Model:
    def __init__(self, env, policy="MlpPolicy"):
        self.env=env
        self.policy=policy
        self.inner_model=PPO(self.policy, self.env, verbose=0,
                            n_steps=N_STEPS, batch_size=BATCH_SIZE)
        self.Loop_Loss=0
        # Tensorboard Writer
        self.writer = SummaryWriter(log_dir='./META_tensorboard_logs')

    def custom_train(self, meta_model):
        """
        Update policy using the currently gathered rollout buffer.
        """
        # Switch to train mode (this affects batch norm / dropout)
        self.inner_model.policy.set_training_mode(True)
        # Compute current clip range
        clip_range = self.inner_model.clip_range(
            self.inner_model._current_progress_remaining)  # type: ignore[operator]
        # Optional: clip range for the value function
        if self.inner_model.clip_range_vf is not None:
            clip_range_vf = self.inner_model.clip_range_vf(
                self.inner_model._current_progress_remaining)  # type: ignore[operator]

        entropy_losses = []
        pg_losses, value_losses = [], []
        clip_fractions = []

        # Do a complete pass on the rollout buffer
        for rollout_data in self.inner_model.rollout_buffer.get(self.inner_model.batch_size):
            actions = rollout_data.actions
            if isinstance(self.inner_model.action_space, spaces.Discrete):
                # Convert discrete action from float to long
                actions = rollout_data.actions.long().flatten()

            # Re-sample the noise matrix because the log_std has changed
            if self.inner_model.use_sde:
                self.inner_model.policy.reset_noise(
                    self.inner_model.batch_size)

            values, log_prob, entropy = self.inner_model.policy.evaluate_actions(
                rollout_data.observations, actions)
            values = values.flatten()
            # Normalize advantage
            advantages = rollout_data.advantages
            # Normalization does not make sense if mini batchsize == 1, see GH issue #325
            if self.inner_model.normalize_advantage and len(advantages) > 1:
                advantages = (advantages - advantages.mean()
                              ) / (advantages.std() + 1e-8)

            # ratio between old and new policy, should be one at the first iteration
            ratio = torch.exp(log_prob - rollout_data.old_log_prob)

            # clipped surrogate loss
            policy_loss_1 = advantages * ratio
            policy_loss_2 = advantages * \
                torch.clamp(ratio, 1 - clip_range, 1 + clip_range)
            policy_loss = -torch.min(policy_loss_1, policy_loss_2).mean()

            # Logging
            pg_losses.append(policy_loss.item())
            clip_fraction = torch.mean(
                (torch.abs(ratio - 1) > clip_range).float()).item()
            clip_fractions.append(clip_fraction)

            if self.inner_model.clip_range_vf is None:
                # No clipping
                values_pred = values
            else:
                # Clip torche difference between old and new value
                # NOTE: torchis depends on torche reward scaling
                values_pred = rollout_data.old_values + torch.clamp(
                    values - rollout_data.old_values, -clip_range_vf, clip_range_vf
                )
            # Value loss using torche TD(gae_lambda) target
            value_loss = F.mse_loss(rollout_data.returns, values_pred)
            value_losses.append(value_loss.item())

            # Entropy loss favor exploration
            if entropy is None:
                # Approximate entropy when no analytical form
                entropy_loss = -torch.mean(-log_prob)
            else:
                entropy_loss = -torch.mean(entropy)

            entropy_losses.append(entropy_loss.item())

            loss = policy_loss + self.inner_model.ent_coef * \
                entropy_loss + self.inner_model.vf_coef * value_loss

            # Optimization step
            self.inner_model.policy.optimizer.zero_grad()
            loss.backward()
            self.inner_model.policy.optimizer.step()

            # Get current state_dict
            current_params = {k: v.clone() for k, v in self.inner_model.policy.state_dict().items()}
            
            # Get gradients
            grad_loss = {k: v.grad.clone() for k, v in self.inner_model.policy.named_parameters()}
            
            # Meta model updates
            alpha, beta = meta_model(torch.tensor([list(current_params.values()), list(grad_loss.values())]))
            
            # Apply updates to parameters
            with torch.no_grad():
                for key in current_params.keys():
                    current_params[key] =beta * current_params[key] - alpha * grad_loss[key]
            print(f"Alpha: {alpha}, Beta:{beta}")

            # Update model parameters with clipped gradient norm
            for name, param in self.inner_model.policy.named_parameters():
                param.grad = torch.clamp(param.grad, -self.inner_model.max_grad_norm, self.inner_model.max_grad_norm)

            # Apply the updates
            self.inner_model.policy.optimizer.step()
            self.Loop_Loss+=loss
    
    def update_base(self, scenario, meta_model, num_inner_updates):
        self.inner_model=PPO(self.policy, self.env, verbose=0,
                            n_steps=N_STEPS,  batch_size=BATCH_SIZE)
        callback=CustomCallback()
        for eps in range(num_inner_updates):
            temp=[[], [], [], [], [], [], []]
            self.inner_model.rollout_buffer.reset()
            # Collect Rollout Buffer
            for x in range(N_STEPS//SIM_TIME):
                print(self.inner_model._last_obs)

                obs = self.env.reset()
                done = False
                
                for day in range(SIM_TIME):
                    action = self.inner_model.predict(obs)
                    next_state, reward, done, info = self.env.step(action[0])
                    print("State_",next_state)
                    print(SIM_TIME * x + day)
                    temp[0].append(obs)
                    temp[1].append(action[0])
                    temp[2].append(reward)
                    temp[3].append(self.inner_model.returns)
                    temp[4].append(self.inner_model.log_probs)
                    temp[5].append(self.inner_model.values)
                    temp[6].append(self.inner_model.advantages)
                    obs = next_state

            self.inner_model.rollout_buffer.obsrtvations
            self.custom_train(meta_model)

        print(rollout_buffer) 

        return np.sum(rollout_buffer.reward)
    
    def log_to_tensorboard(self, iteration, mean_reward, std_reward):
        """
        Logs the metrics to TensorBoard.
        """
        self.writer.add_scalar("Total_Loss", Loop_Loss, iteration)
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
def main():
    # Start timing the computation
    start_time = time.time()
    
    # test_Scenario
    test_scenario = {"Dist_Type": "UNIFORM", "min": 9, "max": 14}
    # Makes environment
    env=GymInterface()
    # Makes Instance
    base_model=Base_Model(env)
    meta_model=Meta_Model()

    for iteartion in range(num_outer_updates):
        # Sample a batch of scenarios
        if len(scenario_distribution) > scenario_batch_size:
            scenario_batch = np.random.choice(
                scenario_distribution, scenario_batch_size, replace=False)
        else:
            scenario_batch = train_scenario_distribution
        
        total_reward=0
        for scenario in scenario_batch:
            print("\n\nTRAINING SCENARIO: ", scenario)
            print("\nOuter Loop: ", env.cur_outer_loop,
                " / Inner Loop: ", env.cur_inner_loop)
            total_reward += base_model.update_base(scenario, meta_model, num_inner_updates)
        
        base_model.log_to_tensorboard(iteration)
        meta_model.update_model(Loop_Loss)
        meta_model.Loop_Loss=0
        torch.save({'model_state_dict': meta_model.state_dict(),
        'optimizer_state_dict': meta_model.optimizer.state_dict(),},
         'model_checkpoint.pth')
main()