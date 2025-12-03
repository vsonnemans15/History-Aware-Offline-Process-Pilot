import types
import sys
sys.modules['d3rlpy.healthcheck'] = types.SimpleNamespace(run_healthcheck=lambda: None)

from MDP_functions import *
from Evaluation import *
import argparse
import math
from sklearn.preprocessing import StandardScaler
import copy
from sklearn.neighbors import NearestNeighbors

import d3rlpy
from d3rlpy.algos.qlearning.cql import DiscreteCQLConfig
from d3rlpy.algos import DiscreteCQL, DiscreteBCQ, DiscreteBC, DQN
from d3rlpy.algos import DiscreteCQLConfig, DQNConfig

import gymnasium as gym
from gym import Env 
from gym import spaces 
import random 
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from collections import Counter
import os
import sys
import pickle
import time
import math
from sklearn.preprocessing import StandardScaler
from scipy.sparse import dok_matrix
from sklearn.preprocessing import LabelEncoder

import math
import random
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

np.random.seed(0)
torch.manual_seed(0)

import argparse

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def parse_args():
    parser = argparse.ArgumentParser(description="DQN training for process monitoring environment")

    parser.add_argument("--dataset", type=str, default="SimBank", help="Dataset to use")
    parser.add_argument("--method", type=str, choices=["CQL", "BC", "BCQ"], default="CQL", help="RL method")
    parser.add_argument("--num_steps", type=int, default=4000000, help="Number of training steps")
    parser.add_argument("--batch_size", type=int, default=100, help="Batch size for training")
    parser.add_argument("--target_update", type=int, default=8000, help="Target update")
    parser.add_argument("--n_hidden1", type=int, default=250, help="Number of hidden neurons")
    parser.add_argument("--n_hidden2", type=int, default=250, help="Number of hidden neurons")
    parser.add_argument("--deep_network", type=str2bool, default=False, help="Deep neural architecture")
    parser.add_argument("--order", type=int, default=6, help="Markov order")
    parser.add_argument("--MDP_model", type=str, choices=["POMDP", "HMDP", "MDP"], default="HMDP", help="Decision-making model class")
    parser.add_argument("--num_steps_ope", type=int, default=1000000, help="Number of training steps")
    return parser.parse_args()


args = parse_args()
dataset = args.dataset
order = args.order # order k 
concatenated_states = True
method = args.method
num_steps = args.num_steps
MDP_model = args.MDP_model
BATCH_SIZE = args.batch_size
target_update = args.target_update
train_fraction = 0.8

device = "cuda" if torch.cuda.is_available() else "cpu"

if torch.cuda.is_available():
    print("CUDA GPU is available")
    print("Number of GPUs:", torch.cuda.device_count())
    print("Current GPU:", torch.cuda.current_device())
    print("GPU Name:", torch.cuda.get_device_name(torch.cuda.current_device()))

endfile = f'final'
class BasicEnv(Env):

    def __init__(self, dataset, order, train_fraction, MDP_model):

        self.dataset = dataset
        self.MDP_model = MDP_model
        self.order = order
        self.df, self.dataset_manager = get_real_data(dataset)
        self.config = DatasetMDP(dataset, self.df, self.dataset_manager)
        self.df = self.config.df
        self.all_cases = self.config.all_cases
        self.all_actions = self.config.all_actions
        self.activity_index = self.config.activity_index
        self.n_actions = self.config.n_actions
        self.action_space = spaces.Discrete(self.n_actions)
      
        self.df = self.config.cut_at_terminal_states(self.df,self.dataset_manager)
        self.df['last_action'] = self.df['action'].astype(str)
        #split into train and test set
        train_size = int(train_fraction * len(self.all_cases))
        self.train_case_ids = self.all_cases[:train_size] 
        self.test_case_ids = self.all_cases[train_size:] 
        self.train_df = self.df[self.df['ID'].isin(self.train_case_ids)].copy()
        self.test_df = self.df[self.df['ID'].isin(self.test_case_ids)].copy()


        self.state_cols,  self.environmental_actions  = define_real_state_cols(dataset, self.dataset_manager)

        if self.dataset == 'SimBank':
            if self.MDP_model == 'HMDP': #removing control flow variables that make the process HMDP
                cols_to_remove = [col for col in self.environmental_actions if col not in ['receive_acceptance', 'receive_refusal']]
                self.state_cols = [col for col in self.state_cols if col not in cols_to_remove]

        self.reward = 0
        self.reward_scale = 10000
        self.unscaled_outcome = 0
        self.case = 0
        self.state = 0
        self.action = 0
        self.costs_dic = self.config.costs_dic
        self.dynamic_cols = self.dataset_manager.dynamic_cat_cols + self.dataset_manager.dynamic_num_cols
        self.dynamic_num_cols = self.dataset_manager.dynamic_num_cols
        
        # scale numeric columns
        self.numeric_cols = self.dataset_manager.static_num_cols + self.dataset_manager.dynamic_num_cols
        self.scaler = StandardScaler()
        self.df[self.numeric_cols] = self.scaler.fit_transform(self.df[self.numeric_cols])
        self.train_df[self.numeric_cols] = self.scaler.fit_transform(self.train_df[self.numeric_cols]) #fit on train set
        self.test_df[self.numeric_cols] = self.scaler.transform(self.test_df[self.numeric_cols]) #transform test set (without fitting to avoid data leakage)
        
         # --- encode categorical columns ---
        self.dataset_manager.dynamic_cat_cols.append('last_action')
        self.categorical_cols = self.dataset_manager.static_cat_cols + self.dataset_manager.dynamic_cat_cols
        
        from sklearn.preprocessing import OneHotEncoder
        self.onehot_cols = []
        self.onehot_encoders = {}
        self.embedding_cols = {}
        self.label_encoders = {}  
    
        for col in self.categorical_cols:
            n_unique = self.train_df[col].nunique()
            if n_unique <= 100:
                # One-hot encode 
                ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore', dtype=np.float32)
                reshaped = self.train_df[[col]] 
                encoded = ohe.fit_transform(reshaped)
                ohe_cols = [f"{col}_{cat}" for cat in ohe.categories_[0]]
                train_df_ohe = pd.DataFrame(encoded, columns=ohe_cols, index=self.train_df.index)
                self.train_df = pd.concat([self.train_df, train_df_ohe], axis=1)
                if col in self.state_cols:
                    self.state_cols.remove(col)
                self.state_cols += ohe_cols
                self.onehot_encoders[col] = ohe  
                self.onehot_cols.append(col)
                
                # Transform testing set using the same encoder
                reshaped_test = self.test_df[[col]]
                encoded_test = ohe.transform(reshaped_test)
                df_ohe_test = pd.DataFrame(encoded_test, columns=ohe_cols, index=self.test_df.index)
                self.test_df = pd.concat([self.test_df, df_ohe_test], axis=1)

            else:
                # Label encode
                le = LabelEncoder()
                self.train_df[col] = le.fit_transform(self.train_df[col])
                self.label_encoders[col] = le
                self.embedding_cols[col] = len(le.classes_)
                self.test_df[col] = self.test_df[col].map(lambda x: le.transform([x])[0] if x in le.classes_ else -1)

        if self.order > 1: # we lagg dynamic state variables (including prior actions) to obtain k-step augmented states
            dynamic_cols_to_lag = []

            # numeric dynamic features
            dynamic_cols_to_lag += [col for col in self.dynamic_num_cols if col in self.state_cols]

            # one-hot dynamic categorical features
            for orig_col in self.dataset_manager.dynamic_cat_cols:
                if orig_col in self.onehot_cols: 
                    oh_cols = [c for c in self.state_cols if c.startswith(orig_col + "_")]
                    dynamic_cols_to_lag += oh_cols
                elif orig_col in self.train_df.columns:  
                    dynamic_cols_to_lag.append(orig_col)

            def add_lagged_features(df, order, cols_to_lag):
                lagged_dict = {}
                for col in cols_to_lag:
                    for lag in range(1, order):
                        lagged_col = f"{col}_lag{lag}"
                        lagged_dict[lagged_col] = (
                            df.groupby("ID")[col]
                            .shift(lag)
                            .fillna(0)  
                        )
                lagged_df = pd.DataFrame(lagged_dict, index=df.index)
                df = pd.concat([df, lagged_df], axis=1)
                return df, list(lagged_dict.keys())

        
            self.train_df, lagged_cols = add_lagged_features(self.train_df, self.order, dynamic_cols_to_lag)
            self.test_df, _ = add_lagged_features(self.test_df, self.order, dynamic_cols_to_lag)

            self.lagged_cols = lagged_cols
            self.state_cols += lagged_cols

        self.penalty_illegal = 1.1 * max(v for v in self.costs_dic.values() if isinstance(v, (int, float)))
        if self.penalty_illegal == 0:
            self.penalty_illegal = 1 

    
    def build_state_with_lags(self, previous_trace, current_event): #for SimBank online evaluation
        """
        Builds the state vector including lagged columns depending on the order.
        Applies one-hot encoding for categorical features at inference.
        """
        state_dict = current_event.copy()
        state_dict['last_action']=state_dict['activity']

        # apply one-hot encoding for categorical columns
        for col in self.onehot_cols:
            ohe = self.onehot_encoders[col]  
            val = str(state_dict.get(col)) 
            value = np.array([[val]], dtype=object)  
            onehot_values = ohe.transform(value).flatten()  
            for i, cat in enumerate(ohe.categories_[0]):
                state_dict[f"{col}_{cat}"] = onehot_values[i]
            if col in state_dict:
                del state_dict[col]

        # Add lagged features
        if self.order > 1:
            for lag in range(1, self.order):
                if len(previous_trace) >= lag:
                    prev_event = previous_trace[-lag]
                    for col in self.dynamic_num_cols:
                        lagged_col = f"{col}_lag{lag}"
                        state_dict[lagged_col] = prev_event.get(col, 0)
                    for col in self.onehot_cols:
                        ohe = self.onehot_encoders[col]
                        val = str(prev_event.get(col, "missing"))
                        value = np.array([[val]], dtype=object)
                        onehot_values = ohe.transform(value).flatten()
                        for i, cat in enumerate(ohe.categories_[0]):
                            lagged_col_name = f"{col}_{cat}_lag{lag}"
                            state_dict[lagged_col_name] = onehot_values[i]
                    
                else:
                    for col in self.dynamic_num_cols:
                        lagged_col = f"{col}_lag{lag}"
                        state_dict[lagged_col] = 0
                    for col in self.onehot_cols:
                        ohe = self.onehot_encoders[col]
                        for i, cat in enumerate(ohe.categories_[0]):
                            lagged_col_name = f"{col}_{cat}_lag{lag}"
                            state_dict[lagged_col_name] = 0

        state_row = [state_dict[col] for col in self.state_cols]
        current_state = np.array(state_row, dtype=np.float32).reshape(1, -1)  

        return current_state
    
env = BasicEnv(dataset, order, train_fraction, MDP_model)

# Constructing custom offline RL traces for d3rlpy algorithm
state_cols = env.state_cols
train_df = env.train_df.copy()  
train_df['action_nr'] = train_df['action'].map(env.activity_index)
train_df['reward'] = 0.0 
train_df['done'] = False

for case_id, case_df in train_df.groupby('ID'): # add rewards and done in df
    cum_cost = 0.0
    reward = 0.0
    granted = False
   
    for idx, row in case_df.iterrows():
        state = row.to_dict()
        action_name = state['action']
        cost_action = env.costs_dic.get(action_name, 0)
        train_df.loc[idx, 'reward'] = -cost_action
        is_terminal, granted_flag = env.config.is_terminal_successful_state(state)
        granted = granted_flag or granted
        
        if is_terminal:
            r = env.config.outcome_function(granted, row[env.state_cols], cost_action, env.scaler, env.numeric_cols, env.state_cols)
            train_df.loc[idx, 'reward'] = r
            train_df.loc[idx, 'done'] = True
            break

observations = []
actions = []
rewards = []
terminals = []
for case_id, case_df in train_df.groupby("ID"): 

    len_case = len(case_df)
    for idx in range(0, len_case-1):
        observations.append(case_df.iloc[idx][state_cols])
        actions.append(case_df.iloc[idx+1]['action_nr'])
        rewards.append(case_df.iloc[idx+1]['reward'])
        terminals.append(case_df.iloc[idx+1]['done'])

        if case_df.iloc[idx+1]['done']:
            break


observations = np.array(observations, dtype=np.float32)
actions = np.array(actions, dtype=np.int64).reshape(-1, 1)
rewards = np.array(rewards, dtype=np.float32)
terminals = np.array(terminals, dtype=bool) #flags indicating whether the case ended after each transition

offline_dataset = d3rlpy.dataset.MDPDataset( #RL traces
    observations=observations,
    actions=actions,
    rewards=rewards,
    terminals=terminals,
)

print(f"============= Offline RL Training with algo {method} ==============")
print(f"Training the {method} agent on {args.dataset} with {num_steps} steps")
print(f" order: {order}, State cols: {env.state_cols}")
print(f"Batch size: {BATCH_SIZE}")

encoder_factory = d3rlpy.models.VectorEncoderFactory(
    hidden_units=[args.n_hidden1, args.n_hidden2],
    activation='relu',
)
import dataclasses
import torch
import torch.nn as nn

from d3rlpy.models.encoders import EncoderFactory
from d3rlpy.preprocessing import StandardRewardScaler

class CustomEncoder(nn.Module):
    def __init__(self, observation_shape, feature_size):
        super().__init__()
        self.feature_size = feature_size
        self.fc1 = nn.Linear(observation_shape[0], args.n_hidden1)
        self.fc2 = nn.Linear(args.n_hidden1, args.n_hidden2)
        self.fc3 = nn.Linear(args.n_hidden2, feature_size)

    def forward(self, x):
        h = torch.relu(self.fc1(x))
        h = torch.relu(self.fc2(h))
        h = torch.relu(self.fc3(h))
        return h

@dataclasses.dataclass()
class CustomEncoderFactory(EncoderFactory):
    feature_size: int

    def create(self, observation_shape):
        return CustomEncoder(observation_shape, self.feature_size)

    @staticmethod
    def get_type() -> str:
        return "custom"
    
if args.deep_network: # for BPIC12 and BPIC17
    config = DiscreteCQLConfig(batch_size=args.batch_size, target_update_interval=target_update,reward_scaler=StandardRewardScaler(),encoder_factory=CustomEncoderFactory(feature_size=64))
    algo = DiscreteCQL(config=config, device=device, enable_ddp=False)

else:
    if method == 'CQL':
        config = DiscreteCQLConfig(batch_size=args.batch_size, target_update_interval=target_update, reward_scaler=StandardRewardScaler(),encoder_factory=encoder_factory)
        algo = DiscreteCQL(config=config, device=device, enable_ddp=False)
    elif method == 'BCQ':
        from d3rlpy.algos import DiscreteBCQConfig
        config = DiscreteBCQConfig()
        algo = DiscreteBCQ(config=config, device=device, enable_ddp=False)

    elif method == 'BC':
        from d3rlpy.algos import DiscreteBCConfig
        config = DiscreteBCConfig()
        algo = DiscreteBC(config=config, device=device, enable_ddp=False)

    else:
        raise ValueError(f"Unknown offline RL method: {method}")

algo.build_with_dataset(offline_dataset)

start_time = time.time()
results = algo.fit(offline_dataset, n_steps=num_steps, save_interval=100, experiment_name=f'{args.dataset}_{method}_{MDP_model}_{order}_N_{num_steps}_batch_{args.batch_size}_nhidden1_{args.n_hidden1}_nhidden2_{args.n_hidden2}_deep_{args.deep_network}_target_{target_update}')
end_time = time.time()
runtime = end_time - start_time 
results_df = pd.DataFrame([{"epoch": e, **m} for e, m in results])

os.makedirs("trained_algo", exist_ok=True)
algo.save_model(f'trained_algo/{args.dataset}_{method}_{MDP_model}_order_{order}_N_{num_steps}_batch_{args.batch_size}_nhidden1_{args.n_hidden1}_nhidden2_{args.n_hidden2}_deep_{args.deep_network}_target_{target_update}.pt')

results = {"method": method,
           "dataset": args.dataset,
           "train_df": train_df,
           "loss_stat_df": results_df,
            'num_steps': num_steps,
            "runtime": runtime,
            "order": order,
            "dataset_size": len(offline_dataset.episodes),
            "train_case_ids":  env.train_case_ids,
            "device": device, 
            "args": vars(args)}

os.makedirs("results_real", exist_ok=True)
with open(f"results_real/{args.dataset}_{method}_{MDP_model}_order_{order}_N_{num_steps}_batch_{args.batch_size}_nhidden1_{args.n_hidden1}_nhidden2_{args.n_hidden2}_deep_{args.deep_network}_target_{target_update}_{endfile}_training.pkl", "wb") as f:
        pickle.dump(results, f)
print("saved")

test_results = {}

if dataset == 'SimBank': 

    test_results = online_evaluation(env, algo, args)
    print("Testing completed")
    os.makedirs("results_real", exist_ok=True)
    with open(f"results_real/{args.dataset}_{method}_{MDP_model}_order_{order}_N_{num_steps}_batch_{args.batch_size}_nhidden1_{args.n_hidden1}_nhidden2_{args.n_hidden2}_deep_{args.deep_network}_target_{target_update}_{endfile}_testing.pkl", "wb") as f:
            pickle.dump(test_results, f)
    print("saved")


from d3rlpy.ope import FQEConfig

print(f"============= Train OPE estimator for {method} policy ==============")

# Get testing RL traces
test_df = env.test_df.copy()  
test_df['action_nr'] = test_df['action'].map(env.activity_index)
test_df['reward'] = 0.0 
test_df['done'] = False

for case_id, case_df in test_df.groupby('ID'): # add rewards and done in df
    cum_cost = 0.0
    reward = 0.0
    granted = False
   
    for idx, row in case_df.iterrows():
        state = row.to_dict()
        action_name = state['action']
        cost_action = env.costs_dic.get(action_name, 0)
        test_df.loc[idx, 'reward'] = -cost_action
        is_terminal, granted_flag = env.config.is_terminal_successful_state(state)
        granted = granted_flag or granted
        
        if is_terminal:
            r = env.config.outcome_function(granted, row[env.state_cols], cost_action, env.scaler, env.numeric_cols, env.state_cols)
            test_df.loc[idx, 'reward'] = r
            test_df.loc[idx, 'done'] = True
            break

observations = []
actions = []
rewards = []
terminals = []
for case_id, case_df in test_df.groupby("ID"):

    len_case = len(case_df)
    for idx in range(0, len_case-1):

        observations.append(case_df.iloc[idx][state_cols])
        actions.append(case_df.iloc[idx+1]['action_nr'])
        rewards.append(case_df.iloc[idx+1]['reward'])
        terminals.append(case_df.iloc[idx+1]['done'])

        if case_df.iloc[idx+1]['done']:
            break


observations = np.array(observations, dtype=np.float32)
actions = np.array(actions, dtype=np.int64).reshape(-1, 1)
rewards = np.array(rewards, dtype=np.float32)
terminals = np.array(terminals, dtype=bool)

test_dataset = d3rlpy.dataset.MDPDataset(
    observations=observations,
    actions=actions,
    rewards=rewards,
    terminals=terminals,
)

if args.deep_network:
    ope_config = FQEConfig(batch_size=args.batch_size, target_update_interval=target_update, reward_scaler=StandardRewardScaler(), encoder_factory=CustomEncoderFactory(feature_size=64))
    fqe = d3rlpy.ope.DiscreteFQE(
            algo=algo,
            config=ope_config,
        )
else: 
    ope_config = FQEConfig(batch_size=args.batch_size, target_update_interval=target_update, encoder_factory=encoder_factory)
    fqe = d3rlpy.ope.DiscreteFQE(
                algo=algo,
                config=ope_config,
            )
    
start_time = time.time()
ope_config = FQEConfig()
fqe = d3rlpy.ope.DiscreteFQE(
    algo=algo,
    config=ope_config,
)

# Train FQE on test cases
results_fqe = fqe.fit(
    dataset=offline_dataset,
    n_steps=args.num_steps_ope,
    n_steps_per_epoch=1000,
    evaluators={
      'average_value': d3rlpy.metrics.AverageValueEstimationEvaluator(test_dataset.episodes),
      'action_match': d3rlpy.metrics.DiscreteActionMatchEvaluator(test_dataset.episodes),
      }

)
print("FQE evaluation completed")
end_time = time.time()
runtime_seconds = end_time - start_time
print(f"FQE evaluation completed in {runtime_seconds:.2f} seconds")

test_results['FQE'] = results_fqe
test_results['FQE_runtime_seconds'] = runtime_seconds
test_results['test_df'] = test_df

print("Testing completed")
os.makedirs("results_real", exist_ok=True)
with open(f"results_real/{args.dataset}_{method}_{MDP_model}_order_{order}_N_{num_steps}_batch_{args.batch_size}_nhidden1_{args.n_hidden1}_nhidden2_{args.n_hidden2}_deep_{args.deep_network}_target_{target_update}_{endfile}_testing.pkl", "wb") as f:
        pickle.dump(test_results, f)
print("saved")

os.makedirs("trained_fqe", exist_ok=True)
fqe.save_model(f'trained_fqe/{args.dataset}_{method}_{MDP_model}_order_{order}_N_{args.num_steps_ope}_batch_{args.batch_size}__nhidden1_{args.n_hidden1}_nhidden2_{args.n_hidden2}_deep_{args.deep_network}_target_{target_update}.pt')
