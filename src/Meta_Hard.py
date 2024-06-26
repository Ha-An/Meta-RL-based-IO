import GymWrapper as gw
import time
import HyperparamTuning as ht  # Module for hyperparameter tuning
from GymWrapper import GymInterface
from config_SimPy import *
from config_RL import *
from log_SimPy import *
from log_RL import *
import torch
import torch.nn as nn
from collections import OrderedDict
from stable_baselines3 import PPO
import learn2learn as l2l
# SB3 모델 로드 및 훈련
env = GymInterface()

# PyTorch 모델 정의
# SB3 모델 로드 및 훈련
# PyTorch 모델 정의
class CustomModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(CustomModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, output_dim)
        self.value_fc1 = nn.Linear(input_dim, 64)
        self.value_fc2 = nn.Linear(64, 64)
        self.value_out = nn.Linear(64, 1)

    def forward(self, x):
        policy_x = torch.relu(self.fc1(x))
        policy_x = torch.relu(self.fc2(policy_x))
        policy_out = self.fc3(policy_x)

        value_x = torch.relu(self.value_fc1(x))
        value_x = torch.relu(self.value_fc2(value_x))
        value_out = self.value_out(value_x)

        return policy_out, value_out

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

# PyTorch 모델 초기화
input_dim = env.observation_space.shape[0]
output_dim = env.action_space.shape[0]
Inner_model=PPO('MlpPolicy', env, verbose=0,n_steps=SIM_TIME)
Outter_model = CustomModel(input_dim, output_dim)


# SB3 매개변수를 PyTorch 모델에 로드
def load_sb3_to_pytorch(sb3_model, torch_model,x):
    # 매핑을 정의합니다 (SB3 -> PyTorch)
    mapping = {
        'mlp_extractor.policy_net.0.weight': 'fc1.weight',
        'mlp_extractor.policy_net.0.bias': 'fc1.bias',
        'mlp_extractor.policy_net.2.weight': 'fc2.weight',
        'mlp_extractor.policy_net.2.bias': 'fc2.bias',
        'mlp_extractor.value_net.0.weight': 'value_fc1.weight',
        'mlp_extractor.value_net.0.bias': 'value_fc1.bias',
        'mlp_extractor.value_net.2.weight': 'value_fc2.weight',
        'mlp_extractor.value_net.2.bias': 'value_fc2.bias',
        'action_net.weight': 'fc3.weight',
        'action_net.bias': 'fc3.bias',
        'value_net.weight': 'value_out.weight',
        'value_net.bias': 'value_out.bias'
    }
    
    if x==0:
        sb3_params = sb3_model.policy.state_dict()
    
    
        torch_params = torch_model.state_dict()
        for sb3_name, torch_name in mapping.items():
            torch_params[torch_name].data=sb3_params[sb3_name].data

        torch_model.state_dict=torch_params
    else:
        sb3_params = sb3_model.policy.state_dict
        torch_params = torch_model.state_dict
        for sb3_name, torch_name in mapping.items():
            torch_params[torch_name].data=sb3_params()[sb3_name].data

        torch_model.state_dict=torch_params 

# PyTorch 매개변수를 SB3 모델에 로드
def load_pytorch_to_sb3(torch_model, sb3_model,x):
    # 매핑을 정의합니다 (PyTorch -> SB3)
    mapping = {
        'fc1.weight': 'mlp_extractor.policy_net.0.weight',
        'fc1.bias': 'mlp_extractor.policy_net.0.bias',
        'fc2.weight': 'mlp_extractor.policy_net.2.weight',
        'fc2.bias': 'mlp_extractor.policy_net.2.bias',
        'value_fc1.weight': 'mlp_extractor.value_net.0.weight',
        'value_fc1.bias': 'mlp_extractor.value_net.0.bias',
        'value_fc2.weight': 'mlp_extractor.value_net.2.weight',
        'value_fc2.bias': 'mlp_extractor.value_net.2.bias',
        'fc3.weight': 'action_net.weight',
        'fc3.bias': 'action_net.bias',
        'value_out.weight': 'value_net.weight',
        'value_out.bias': 'value_net.bias'
    }
    if x==0:
        torch_params = torch_model.state_dict
        sb3_params = sb3_model.policy.state_dict
        for torch_name, sb3_name in mapping.items():
            sb3_params()[sb3_name].data=torch_params()[torch_name].data

        sb3_model.policy.state_dict=sb3_params
    else:
        torch_params = torch_model.state_dict
        sb3_params = sb3_model.policy.state_dict
        for torch_name, sb3_name in mapping.items():
            sb3_params()[sb3_name].data=torch_params[torch_name].data

        sb3_model.policy.state_dict=sb3_params
        
# PyTorch 매개변수를 SB3 모델에 로드
load_pytorch_to_sb3(Outter_model, Inner_model,0)

# SB3 매개변수를 PyTorch 모델에 로드
load_sb3_to_pytorch(Inner_model, Outter_model,0)

dist_info=Make_Task()
opt = torch.optim.Adam(Outter_model.parameters(), lr=0.001)
x=0
for iteration in range(100):
    iteration_loss = 0.0
    iteration_reward = 0.0
    tasks=[dist_info[random.randint(0,len(dist_info)-1)] for x in range(1)]
    load_pytorch_to_sb3(Outter_model, Inner_model,1)
    for task in tasks:  # Samples a new config
        env.Dist_info=task
        env.reset()

        # Fast Adapt
        for step in range(5):
            Inner_model.learn(total_timesteps=SIM_TIME)

        done=False
        obs=env.reset()
        while done:
            action=Inner_model.predict(obs)
            obs,reward,done,_=Inner_model.step(action)
        Reward =torch.tensor(REWARD_LOG,dtype=torch.float32)
        loss=Reward.mean()
        iteration_loss += loss.item()
        iteration_reward -= loss.item()

    # Print statistics
    print('\nIteration', iteration)
    adaptation_reward = iteration_reward / 5
    adaptation_loss=iteration_loss/5
    print('adaptation_reward', adaptation_reward)
    print('adaptation_loss', adaptation_reward)
    load_sb3_to_pytorch(Inner_model, Outter_model,1)
    opt.zero_grad()
    torch.tensor(adaptation_loss,dtype=torch.float32).backward()
    opt.step()



''''
def build_model():
    if RL_ALGORITHM == "DQN":
        model = DQN("MlpPolicy", env, verbose=0)
        # model = DQN("MlpPolicy", env, learning_rate=BEST_PARAMS['learning_rate'], gamma=BEST_PARAMS['gamma'],
        #             batch_size=BEST_PARAMS['batch_size'], verbose=0)
    elif RL_ALGORITHM == "DDPG":
        model = DQN("MlpPolicy", env, verbose=0)
        # model = DDPG("MlpPolicy", env, learning_rate=BEST_PARAMS['learning_rate'], gamma=BEST_PARAMS['gamma'],
        #              batch_size=BEST_PARAMS['batch_size'], verbose=0)
    elif RL_ALGORITHM == "PPO":
        model = PPO("MlpPolicy", env, verbose=0, n_steps=SIM_TIME)
        # model = PPO("MlpPolicy", env, learning_rate=BEST_PARAMS['learning_rate'], gamma=BEST_PARAMS['gamma'],
        #             batch_size=BEST_PARAMS['batch_size'], n_steps=SIM_TIME, verbose=0)
        print(env.observation_space)
    return model



def export_report(inventoryList):
    for x in range(len(inventoryList)):
        for report in DAILY_REPORTS:
            export_Daily_Report.append(report[x])
    daily_reports = pd.DataFrame(export_Daily_Report)
    daily_reports.columns = ["Day", "Name", "Type",
                         "Start", "Income", "Outcome", "End"]
    daily_reports.to_csv("./Daily_Report.csv")
'''



# Optionally render the environment
env.render()
