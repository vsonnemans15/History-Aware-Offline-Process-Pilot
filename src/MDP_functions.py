import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from collections import Counter
import sys
import pickle
import time
import math
from scipy.sparse import dok_matrix
from sklearn.preprocessing import LabelEncoder
import os
import gymnasium as gym
from gym import Env 
from gym import spaces 

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from dataset_manager.DatasetManager import *

np.random.seed(0)


class DatasetMDP:
    def __init__(self, dataset, df, dataset_manager):
        self.dataset = dataset
        self.df = df
        self.control_flow_var = dataset_manager.control_flow_var #includes env actions + other control flow variables
        self.environmental_actions = dataset_manager.environmental_actions
        self.control_flow_var_incremental = dataset_manager.control_flow_var_incremental
        self.control_flow_var_binary = dataset_manager.control_flow_var_binary
        self.control_flow_var_attribute = dataset_manager.control_flow_var_attribute
        if self.dataset != 'hospital_billing_2':
            self.df = self.df[~self.df['action'].isin(['archive_application', 'start'])]
        if self.dataset == 'SimBank': #add the interest rate in the action space
            mask = self.df["action"] == "calculate_offer"
            self.df.loc[mask, "action"] = self.df.loc[mask, "interest_rate"]
        self.df, self.all_cases, self.all_actions, self.activity_index, self.n_actions = self.remove_env_actions(self.df)
       
        self.costs_dic = self.define_costs_from_log()
        self.win_action, self.loss_actions = self.define_terminal_actions()
    
    def cut_at_terminal_states(self, df, dataset_manager):
        """
        Cut traces at the first terminal state per case,
        but also keep the last row of the case (e.g., archive_application).
        """
        def cut_case(case_df):
            for idx, row in case_df.iterrows():
                state = row.to_dict()
                is_terminal, _ = self.is_terminal_successful_state(state)
                if is_terminal:
                    # Keep up to terminal + last row of the case
                    last_idx = case_df.index[-1]
                    if idx == last_idx:
                        return case_df.loc[:idx]  # terminal is already last
                    else:
                        return pd.concat([case_df.loc[:idx], case_df.loc[[last_idx]]])
            return case_df  # no terminal found → return full trace
     
        df = df.groupby("ID", group_keys=False).apply(cut_case).reset_index(drop=True)
        df = df.sort_values(by=['ID', dataset_manager.timestamp_col]).reset_index(drop=True)
        return df
    

    def remove_env_actions(self, df):
        for env_activity in self.control_flow_var: #control_flow_var_attribute remain the same
            if env_activity in self.environmental_actions or env_activity in self.control_flow_var_binary:
                print("Transforming environmental or control flow action as binary state var:", env_activity)
                # Create a binary column: 1 if the action occurs at this row
                df[env_activity] = (df['action'] == env_activity).astype(int)
                # Compute cumulative max per case to keep 1 after first occurrence
                df[env_activity] = df.groupby('ID')[env_activity].cummax()
            elif env_activity in self.control_flow_var_incremental:
                print("Transforming control flow action as incremental state var:", env_activity)
                # Create an incremental column: count occurrences of the action per case
                df[env_activity] = (df['action'] == env_activity).astype(int)
                df[env_activity] = df.groupby('ID')[env_activity].cumsum()
        
        #remove environmental actions by replacing them with the previous action in the trace
        # Boolean mask where actions are control-flow
        mask = df['action'].isin(self.environmental_actions)
        prev_values = df[['action', 'last_action']].shift(1)
        # Assign previous row values to current row where mask is True
        df.loc[mask, ['action', 'last_action']] = prev_values.loc[mask]

        # Drop the previous rows (rows before the masked ones)
        prev_idx = mask.shift(-1, fill_value=False)
        df = df.loc[~prev_idx].reset_index(drop=True)

        all_cases = df['ID'].unique()
        
        if self.dataset == 'hospital_billing_2': 
            df['isCancelled'] = df['isCancelled'].map({False: 0, True: 1})


        all_actions = df["action"].unique() 
        activity_index = {activity: idx for idx, activity in enumerate(all_actions)}
        n_actions = len(all_actions)
        df['last_action'] = df['action'].map(activity_index)
        
        return df, all_cases, all_actions, activity_index, n_actions



    def define_costs_from_log(self):
        if self.dataset == 'SimBank':
            return {
                "initiate_application": 0,
                    "start_standard": 10,
                    "start_priority": 5000,
                    "validate_application": 20,
                    "contact_headquarters": 1000,
                    "skip_contact": 0,
                    "email_customer": 10,
                    "call_customer": 20,
                    "calculate_offer": 400,
                    0.07: 400,
                    0.08: 400,
                    0.09: 400,
                    "cancel_application": 30,
                    "receive_acceptance": 10,
                    "receive_refusal": 10,
                    "stop_application": 0
                }
        

        if self.dataset in ['hospital_billing_2','bpic2012_accepted', 'bpic2017_accepted']:
            unique_activities = self.df['action'].unique()
            return {
                act: 0.0 if act in ["start", "archive_application"] else 10.0
                for act in unique_activities
            }
        if self.dataset in ['sepsis_cases_1', 'sepsis_cases_2']:
            return {
                "start": 0,
                'archive_application': 0,
                "ER Registration": 150,
                "Leucocytes": 25,
                "CRP": 20,
                "LacticAcid": 35,
                "ER Triage": 250,
                "ER Sepsis Triage": 350,
                "IV Liquid": 75,
                "IV Antibiotics": 200,
                "Admission NC": 2000,   
                "Release A": 50,
                "Admission IC": 7000,  
                "Release B": 50,
                "Release E": 50,
                "Release C": 50,
                "Release D": 50,
            }
        
    def is_terminal_successful_state(self, state): 
        """
        Determine whether the current state is terminal and whether it is successful (granted).
        """
        is_terminal = False
        is_successful = False

        # ----- Action-based terminal states ----- 
        if self.dataset in ['bpic2017_accepted', 'bpic2012_accepted']:
            last_action = int(self.activity_index[state.get('action')])
            terminal_actions = self.win_action + self.loss_actions
            if last_action in terminal_actions:
                is_terminal = True
                is_successful = last_action in self.win_action  # granted only if win_action
        elif self.dataset == 'SimBank':
            last_action = int(self.activity_index[state.get('action')])
            if state.get('receive_acceptance')==1:
                is_terminal, is_successful = True, True
            elif state.get('receive_refusal') ==1:
                is_terminal, is_successful = True, False
            elif last_action == self.activity_index['cancel_application']:
                is_terminal, is_successful = True, False

        elif self.dataset == 'sepsis_cases_2':
            last_action = int(self.activity_index[state.get('action')])
            terminal_actions = self.win_action + self.loss_actions
            if last_action in terminal_actions:
                is_terminal = True
                is_successful = last_action in self.win_action  # granted only if win_action
        # ----- Attribute-based terminal states -----
        elif self.dataset == 'sepsis_cases_1':
            last_action = int(self.activity_index[state.get('action')])
            recent_release = state.get('recent_release', 0)
            return_ER = state.get('Return ER')
            if last_action in [self.activity_index['Release A'], self.activity_index['Release B'], self.activity_index['Release C'], self.activity_index['Release D'], self.activity_index['Release E']]:
                is_terminal = True
                if return_ER == 1 and recent_release == 0: # Successful if Return ER happened within 28 days
                     is_successful = True
                elif return_ER == 0: # Successful if no Return ER
                     is_successful = True                    
        
        elif self.dataset == 'hospital_billing_2':
            last_action = int(self.activity_index[state.get('action')])
            cancellation_status = state.get('isCancelled')
            if cancellation_status==1:
                is_terminal, is_successful = True, False
            elif last_action == self.activity_index['archive_application'] and cancellation_status==0:
                is_terminal, is_successful = True, True

        return is_terminal, is_successful

        
    def define_terminal_actions(self):
        if self.dataset == 'bpic2017_accepted':
            win = [self.activity_index['A_Pending']]
            loss = [self.activity_index['A_Cancelled'],
                    self.activity_index['A_Denied']]
            
        elif self.dataset == 'bpic2012_accepted':
            win = [self.activity_index['A_APPROVED-COMPLETE']]
            loss = [self.activity_index['A_CANCELLED-COMPLETE'],
                    self.activity_index['A_DECLINED-COMPLETE']]
            

        elif self.dataset == 'sepsis_cases_1':
            win = [self.activity_index['Release A'], self.activity_index['Release B'], self.activity_index['Release C'], self.activity_index['Release D'], self.activity_index['Release E']]
            loss = []
             
        elif self.dataset == 'sepsis_cases_2':
            win = [self.activity_index['Release A'], self.activity_index['Release B'], self.activity_index['Release C'], self.activity_index['Release D'], self.activity_index['Release E']]
            loss = [self.activity_index['Admission IC']]
             
   
        elif self.dataset == 'hospital_billing_2':
            win = [] #need to go until the end of trace to determine if success
            loss = [self.activity_index['DELETE']]

 
        elif self.dataset == 'SimBank':
            win = []
            loss = [self.activity_index['cancel_application']]
            
        else:
            win, loss = [], []
        return win, loss
    
    def binary_outcome_function(self, granted):
        if granted: 
            r = 1000
        else:
            r = 0
        return r

    def outcome_function_unscaled(self, granted, current_event, cum_cost):
        if self.dataset == 'SimBank':
            if granted:
                #apply inverse transform only on numeric columns
                dummy = pd.DataFrame([current_event]).copy()
                unscaled_event = dummy.iloc[0].to_dict()

                i = unscaled_event["interest_rate"]
                A = unscaled_event["amount"]
                risk_factor = (10 - unscaled_event['est_quality']) / 200 
                risk_free_rate = 0.03 # around the 10 year Belgian government bond yield 16/11/2023, also accounts for inflation
                df = risk_free_rate + risk_factor
                n = 10 # number of years
                future_earnings = A * (1 + i)**n
                discount_future_earnings = future_earnings / (1 + df)**n
                exp_profit = discount_future_earnings - cum_cost - A - 100
                return exp_profit
            else:
                return -cum_cost - 100

    def outcome_function(self, granted, current_event, cum_cost, scaler, numeric_cols, state_cols):
        # dataset-specific reward
        if self.dataset == 'SimBank':
            if granted:
                #apply inverse transform only on numeric columns
                dummy = pd.DataFrame([current_event]).copy()
                unscaled_numeric = scaler.inverse_transform(dummy[numeric_cols])
                dummy.loc[:, numeric_cols] = unscaled_numeric
                unscaled_event = dummy.iloc[0].to_dict()

                i = unscaled_event["interest_rate"]
                A = unscaled_event["amount"]
                risk_factor = (10 - unscaled_event['est_quality']) / 200 
                risk_free_rate = 0.03 # around the 10 year Belgian government bond yield 16/11/2023, also accounts for inflation
                df = risk_free_rate + risk_factor
                n = 10 # number of years
                future_earnings = A * (1 + i)**n
                discount_future_earnings = future_earnings / (1 + df)**n
                exp_profit = discount_future_earnings - cum_cost - A - 100
                return exp_profit
            else:
                return -cum_cost - 100

        elif self.dataset == 'bpic2017_accepted' and granted:
            dummy = pd.DataFrame([current_event])[state_cols].copy()
            unscaled_numeric = scaler.inverse_transform(dummy[numeric_cols])
            dummy.loc[:, numeric_cols] = unscaled_numeric
            unscaled_event = dummy.iloc[0].to_dict()

            A = unscaled_event["OfferedAmount"]
            # Determine interest rate by class
            if A <= 6000:
                r = 0.16
            elif A <= 15000:
                r = 0.18
            else:
                r = 0.20
            interest = A * r
            return interest - cum_cost
        
        elif self.dataset == 'bpic2012_accepted' and granted:
            dummy = pd.DataFrame([current_event])[state_cols].copy()
            unscaled_numeric = scaler.inverse_transform(dummy[numeric_cols])
            dummy.loc[:, numeric_cols] = unscaled_numeric
            unscaled_event = dummy.iloc[0].to_dict()
            A = unscaled_event["AMOUNT_REQ"]
            # Determine interest rate by class
            if A <= 6000:
                r = 0.16
            elif A <= 15000:
                r = 0.18
            else:
                r = 0.20
            interest = A * r
            return interest - cum_cost

        elif self.dataset in ['sepsis_cases_1', 'sepsis_cases_2', 'sepsis_cases_3']:
            if granted:  # successful discharge
                health_benefit = 10000  # proxy survival value
                return health_benefit - cum_cost
            else:
                return -cum_cost

        elif self.dataset == 'hospital_billing_2' and granted:
                revenue = 1000
                return revenue - cum_cost
   
        else: #for all datasets
            return -cum_cost 

def get_real_data(dataset):
      
        dataset_manager = DatasetManager(dataset)
        df = dataset_manager.read_dataset()
        df = df.rename(columns={dataset_manager.activity_col: "action", 
                        dataset_manager.case_id_col: "ID"})
        #dataset_manager.dynamic_cat_cols.remove(dataset_manager.activity_col)

        all_cases = df['ID'].unique()
        all_actions = df["action"].unique() 

        activity_index = {activity: idx for idx, activity in enumerate(all_actions)}
        n_actions = len(all_actions)
        df['last_action'] = df['action'].map(activity_index)
        
        return df, dataset_manager 

def define_real_state_cols(dataset, dataset_manager):

    dataset_manager.dynamic_cat_cols.remove(dataset_manager.activity_col)

    for col in ['hour', 'weekday', 'month', 'timesincemidnight', 
            'timesincelastevent', 'event_nr', 'open_cases']:
        if col in dataset_manager.dynamic_num_cols:
            dataset_manager.dynamic_num_cols.remove(col)

    if dataset != 'traffic_fines_1' and 'timesincecasestart' in dataset_manager.dynamic_num_cols:
        dataset_manager.dynamic_num_cols.remove('timesincecasestart')
    
    control_flow_var = dataset_manager.control_flow_var
    state_cols_simulation = ['last_action'] + dataset_manager.dynamic_cat_cols + dataset_manager.dynamic_num_cols + dataset_manager.static_cat_cols + dataset_manager.static_num_cols + control_flow_var
    
    return state_cols_simulation, control_flow_var

