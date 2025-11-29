print("script started", flush=True)
import argparse
import pandas as pd
import numpy as np
import ctypes.util
import os, sys
import pickle

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

from MDP_functions_updated import *

from test_func._core_test_fun import *

from _DGP_TIGER import *

os.environ["OMP_NUM_THREADS"] = "1"

print("Import DONE!")
print("n_cores = ", n_cores)

# === Parse command-line arguments ===
parser = argparse.ArgumentParser(description='Run MDP Real Evaluation')
parser.add_argument('--T', type=int, default=None)
parser.add_argument('--n_cases', type=int, default=20)
parser.add_argument('--max_lag', type=int, default=10)
parser.add_argument('--L', type=int, default=3)
parser.add_argument('--Q', type=int, default=10)
parser.add_argument('--dataset', type=str, default='bpi2017')
parser.add_argument('--n_samples', type=int, default=200) #number of test runs
parser.add_argument('--rep_offset', type=int, default=0)
parser.add_argument('--MDP_model', type=str, default='Unknown')
args = parser.parse_args()

np.random.seed(230)  # Set a fixed seed for reproducibility


class BasicEnv(Env):

    def __init__(self, dataset, order, train_fraction):

        self.dataset = dataset
        self.order = order
        self.df, self.dataset_manager = get_real_data(dataset)
        self.config = DatasetMDP(dataset, self.df, self.dataset_manager)
        self.df = self.config.df
        self.all_cases = self.config.all_cases
        self.all_actions = self.config.all_actions
        self.activity_index = self.config.activity_index
        self.n_actions = self.config.n_actions
      
        self.df = self.config.cut_at_terminal_states(self.df,self.dataset_manager)
        
        #self.df['action_neuron'], self.action_mapping = pd.factorize(self.df['action'])
        self.df['last_action'] = self.df['action']
        self.df['action_nr'] = self.df['action'].map(self.activity_index)
        #split into train and test set
        train_size = int(train_fraction * len(self.all_cases))
        self.train_case_ids = self.all_cases[:train_size] 
        self.test_case_ids = self.all_cases[train_size:] 
        self.train_df = self.df[self.df['ID'].isin(self.train_case_ids)].copy()
        self.test_df = self.df[self.df['ID'].isin(self.test_case_ids)].copy()


        self.state_cols,  self.environmental_actions  = define_real_state_cols(dataset, self.dataset_manager)
        self.dataset_manager.dynamic_cat_cols.append('last_action')
        self.reward = 0
        self.reward_scale = 10000
        self.unscaled_outcome = 0
        self.case = 0
        self.state = 0
        self.action = 0
        self.costs_dic = self.config.costs_dic
        self.dynamic_cols = self.dataset_manager.dynamic_cat_cols + self.dataset_manager.dynamic_num_cols
        self.dynamic_num_cols = self.dataset_manager.dynamic_num_cols
        
        self.numeric_cols = self.dataset_manager.static_num_cols + self.dataset_manager.dynamic_num_cols
        self.categorical_cols = self.dataset_manager.static_cat_cols + self.dataset_manager.dynamic_cat_cols
        
        self.onehot_cols = []
        self.embedding_cols = {}
        self.label_encoders = {}  # store label encoders for high-cardinality columns
    
        for col in self.categorical_cols:
            n_unique = self.train_df[col].nunique()
            if n_unique <= 100:
                # One-hot encode using pandas get_dummies
                dummies_train = pd.get_dummies(self.train_df[col], prefix=col, dtype=np.float32)
                self.train_df = pd.concat([self.train_df, dummies_train], axis=1)
                self.state_cols.remove(col)
                self.state_cols += list(dummies_train.columns)
                self.onehot_cols.append(col)
                
                dummies_test = pd.get_dummies(self.test_df[col], prefix=col, dtype=np.float32)
                for c in dummies_train.columns:
                    if c not in dummies_test:
                        dummies_test[c] = 0.0  # add missing columns with zeros
                self.test_df = pd.concat([self.test_df, dummies_test[dummies_train.columns]], axis=1)
                
            else:
                # Encode as embedding indices using LabelEncoder fitted on train only
                le = LabelEncoder()
                self.train_df[col] = le.fit_transform(self.train_df[col])
                self.label_encoders[col] = le
                self.embedding_cols[col] = len(le.classes_)
                self.test_df[col] = self.test_df[col].map(lambda x: le.transform([x])[0] if x in le.classes_ else -1)

        self.penalty_illegal = 1.1 * max(v for v in self.costs_dic.values() if isinstance(v, (int, float)))
        if self.penalty_illegal == 0:
            self.penalty_illegal = 1 

def prepare_testmdp_data(df, T, state_cols):
    data = []
    action_col = 'action_nr' #encoded action
    trajectory_col = 'ID'
    feature_cols = [col for col in state_cols if col != action_col]

    for traj_id, traj_df in df.groupby(trajectory_col):
        
        if len(traj_df) < T: #we loose a row when creating data
            continue  # Skip trajectories that don't have exactly T+1 actions
        traj_df = traj_df.tail(T)  # Truncate to min_len

        X = traj_df[feature_cols].to_numpy()[:-1]

        # Ensure A is 2D: reshape if scalar actions
        A_raw = traj_df[action_col].to_numpy()[1:]
        R_raw = traj_df['reward'].to_numpy()[1:]
        if np.ndim(A_raw[0]) == 0:
            A = A_raw.reshape(-1, 1)
        else:
            A = np.stack(A_raw)
        
        if np.ndim(R_raw[0]) == 0:
            R = R_raw.reshape(-1, 1)
        else:
            R = np.stack(R_raw)

        data.append([X, A, R])

    return data
 
order = 1
train_fraction = 0.8
env = BasicEnv(args.dataset, order, train_fraction)
df = env.train_df 
df['action_nr'] = df['action'].map(env.activity_index)
df['reward'] = 0.0 
df['done'] = False
for case_id, case_df in df.groupby('ID'): # add rewards and done in df
    cum_cost = 0.0
    reward = 0.0
    granted = False
   
    for idx, row in case_df.iterrows():
        state = row.to_dict()
        action_name = state['action']
        cost_action = env.costs_dic.get(action_name, 0)
        df.loc[idx, 'reward'] = -cost_action
        is_terminal, granted_flag = env.config.is_terminal_successful_state(state)
        granted = granted_flag or granted
        
        if is_terminal:
            r = env.config.outcome_function_unscaled(granted, row[env.state_cols], cum_cost)
            df.loc[idx, 'reward'] = r
            df.loc[idx, 'done'] = True
            break

state_cols = env.state_cols

if args.T is None :
    T = df.groupby('ID').size().mode()[0] 
else: 
    T = args.T

if args.dataset == 'SimBank' and args.MDP_model == 'HMDP':
    state_cols = [col for col in state_cols if col not in ['cum_cost', 'elapsed_time', 'noc', 'nor', 'skip', 'contact_hq']]
elif args.dataset == 'SimBank' and args.MDP_model == 'MDP':
    state_cols = [col for col in state_cols if col not in ['cum_cost', 'elapsed_time']] #keep control flow variables 'noc', 'nor', 'skip', 'contact_hq'
    

print(state_cols)
print(f"{args.dataset}_{args.MDP_model}_CV_T_{T}_n_cases_{args.n_cases}_L_{args.L}_Q_{args.Q}_n_samples_{args.n_samples}")
df_trajectories = prepare_testmdp_data(df, T, state_cols) #cutting tractories to length T


data = [df_trajectories[i] for i in np.random.choice(len(df_trajectories), size=args.n_cases, replace=False)] #select n_cases randomly
df_norm = normalize(data.copy()) #normalize the data
results = {
    "dataset": args.dataset,
    "T": T,
    "n_cases": args.n_cases,
    "L": args.L,
    "Q": args.Q,
    "n_samples": args.n_samples,
    "state_cols": state_cols,
    "quick_rejection_rates": {},
    "p_values": {}
}

# === Config class ===
class Setting:
    def __init__(self):
        self.T = data[0][0].shape[0] #we loose one observation when preparing the data
        self.x_dims = data[0][0].shape[1] #num of features
        self.N = len(data) #number of cases
        self.L = args.L
        self.Q = args.Q
        self.show = False

config = Setting()
print(config.T)


def one_time(seed = 1, J = 1, 
                     N = 100, T = 20, T_def = 0, 
                     B = 100, Q = 10, 
                     paras = [100, 3,20], weighted = True, include_reward = False,
                     method = "QRF"):
    """
    include_reward: if include reward to our test
    T_def:
        0: length = T with always listen
        1: truncation
    T: the final length
    """
    
    np.random.seed(seed) #changing the seed so that we have different data each time
    df_sel = [df_trajectories[i] for i in np.random.choice(len(df_trajectories), size=N, replace=False)] #select N cases randomly
    MDPs = normalize(df_sel.copy())
    #print(MDPs)

    N = len(MDPs)
    ### Calculate
    if paras == "CV_once":
        return lam_est(data = MDPs, J = J, B = B, Q = Q, L=args.L, paras = paras, include_reward = include_reward,
                  fixed_state_comp = None, method = method)
    return test(data = MDPs, J = J, B = B, Q = Q, L=args.L, paras = paras, #print_time = print_time,
                include_reward = include_reward, fixed_state_comp = None, method = method)


def one_setting_one_J(rep_times = 10, J = 1, 
                      N = 100, T = 20, 
                      B = 100, Q = 10,
                      include_reward = False, mute = True,
                      paras = "CV_once", init_seed = 0, parallel = True, method = "QRF"):
    if paras == "CV_once":
        paras = one_time(seed = 0, J = J, 
                         N = N, T = T, B = B, Q = Q,
                        paras = "CV_once",
                       include_reward = include_reward, method = method)
        print("CV paras:", paras)
    
    def one_test(seed):
        return one_time(seed = seed, J = J, 
                     N = N, T = T, B = B, Q = Q,
                     include_reward = include_reward,
                     paras = paras, method = method)
    p_values = parmap(one_test,range(init_seed, init_seed + rep_times), parallel)
    if not mute:
        print("rejection rates are:", rej_rate_quick(p_values))
    return p_values
print("Import DONE!")

print("n_cores = ", n_cores)

test_lags = list(range(1, args.max_lag+1))
print(f"Max test lag: {test_lags}")

for J in test_lags:
            print("lag = ", J)
            p_values = one_setting_one_J(rep_times = args.n_samples, J = J, 
                              N = args.n_cases, T = config.T,      #config.T = args.T-1 (we loose one obs when preparing data)
                              B = 100, Q = args.Q,
                              include_reward = False, mute = False,
                              paras = "CV_once", init_seed = 4000, parallel = n_cores, method = "QRF") #230
           
            rejection_rate = rej_rate_quick(p_values)
            print("p_values:", p_values)
            results["quick_rejection_rates"][J] = rejection_rate  # ensure serializable
            print("p_values:", p_values)
            results["p_values"][J] = p_values  # store raw list

            package_path = os.path.abspath(os.getcwd())
            output_dir = os.path.join(package_path, "test_results")
            os.makedirs(output_dir, exist_ok=True)
            with open(os.path.join(output_dir, f"{args.dataset}_{args.MDP_model}_CV_T_{T}_n_cases_{args.n_cases}_L_{args.L}_Q_{args.Q}_n_samples_{args.n_samples}_head_reward.pkl"), "wb") as f:
                pickle.dump(results, f)

print("Saving to:", output_dir)
