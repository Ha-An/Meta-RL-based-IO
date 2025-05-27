import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.distributions import Categorical
from config_RL import *
import time
from PPO import *
BUFFER_SIZE = 100000
BATCH_SIZE = 20  
LEARNING_RATE = 0.00005
GAMMA = 0.99
CLIP_EPSILON = 0.2
UPDATE_STEPS = 10  
GAE_LAMBDA = 0.95
ENT_COEF = 0.0
VF_COEF = 0.5
MAX_GRAD_NORM = 0.5


class outer_loop(PPOAgent):
    def update(self, inner_model):
        """
        Performs PPO update using stored experience.

        This function processes stored transitions, computes advantages,
        and updates the policy and value networks using PPO loss.
        """
        '''
        states, actions, rewards, next_states, dones, log_probs_old = zip(*self.memory)
        states = torch.tensor(np.array(states), dtype=torch.float32, device=self.device)
        actions = torch.tensor(np.array(actions), dtype=torch.long, device=self.device)
        rewards = torch.tensor(np.array(rewards), dtype=torch.float32, device=self.device)
        next_states = torch.tensor(np.array(next_states), dtype=torch.float32, device=self.device)
        dones = torch.tensor(np.array(dones), dtype=torch.float32, device=self.device)
        log_probs_old = torch.tensor(np.array(log_probs_old), dtype=torch.float32, device=self.device)
        '''
        
        inner_states, inner_actions, inner_rewards, inner_next_states, inner_dones, inner_log_probs = zip(*inner_model.memory)
        inner_states = torch.tensor(np.array(inner_states), dtype=torch.float32, device=self.device)
        inner_actions = torch.tensor(np.array(inner_actions), dtype=torch.long, device=self.device)
        inner_rewards = torch.tensor(np.array(inner_rewards), dtype=torch.float32, device=self.device)
        inner_next_states = torch.tensor(np.array(inner_next_states), dtype=torch.float32, device=self.device)
        inner_dones = torch.tensor(np.array(inner_dones), dtype=torch.float32, device=self.device)
        inner_log_probs = torch.tensor(np.array(inner_log_probs), dtype=torch.float32, device=self.device)
        
        '''
        _, values = self.policy(inner_states)
        _, next_values = self.policy(inner_next_states)
        not_dones = (1 - dones).unsqueeze(1)
        next_values = (next_values * not_dones).clone()
        '''
        # 경험에서 가져와야할 모든 자료는 inner_model의 데이터를 사용하되, action에 관한 경험은 outer에서 가져옴옴
        _, inner_values = inner_model.model.policy(inner_states)
        _, inner_next_values = inner_model.model.policy(inner_next_states)
        inner_not_dones = (1 - inner_dones).unsqueeze(1)
        inner_next_values = (inner_next_values * inner_not_dones).clone()

        advantages = self._compute_gae(inner_rewards, inner_values.detach().squeeze(), self.gamma, self.gae_lambda)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        inner_value_target = inner_rewards + self.gamma * inner_next_values.view(-1).detach()

        batch_size = BATCH_SIZE  
        dataset_size = len(inner_states)
        indices = np.arange(dataset_size)
        np.random.shuffle(indices)

        start_time = time.time()
        for _ in range(self.update_steps):
            for i in range(0, dataset_size, batch_size):
                batch_indices = indices[i : i + batch_size]
                batch_states = inner_states[batch_indices]
                batch_actions = inner_actions[batch_indices]
                batch_advantages = advantages[batch_indices]
                batch_innner_log_probs = inner_log_probs[batch_indices].detach().clone()
                batch_inner_value_target = inner_value_target[batch_indices].detach().clone()

                action_probs, values_new = self.policy(batch_states)
                
                log_probs_new = []
                for j, dist in enumerate(action_probs):
                    categorical_dist = Categorical(dist)
                    log_probs_new.append(categorical_dist.log_prob(batch_actions[:, j]))
                log_probs_new = torch.sum(torch.stack(log_probs_new), dim=0)
                
                ratio = torch.exp(log_probs_new - batch_innner_log_probs)
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()

                value_loss = nn.MSELoss()(values_new.view(-1), batch_inner_value_target)
                
                entropy = torch.stack([
                    Categorical(dist).entropy().mean() for dist in action_probs
                ]).mean()
                entropy_loss = -entropy
                
                loss = policy_loss + self.ent_coef * entropy_loss + self.vf_coef * value_loss
                
                self.optimizer.zero_grad()
                loss.backward(retain_graph=True)  
                nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=0.5)
                self.optimizer.step()
                
                self.learn_time=time.time()-start_time
        
        self.clip_epsilon = max(0.1, self.clip_epsilon * 0.995)
        
        return self.learn_time
