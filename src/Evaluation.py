from datetime import datetime
from MDP_functions import *
from SimBank.SimBank.simulation import *
import math 
from itertools import product
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

np.random.seed(0)
torch.manual_seed(0)

def online_evaluation(offline_env, algo, args):

    print("================== Online Evaluation with algo {method} =============================")

    with open("dataset_params.pkl", "rb") as f: #same parameters as SimBank paper
        dataset_params = pickle.load(f)
    dataset_params["simulation_start"] = datetime(2024, 3, 20, 8, 0)

    online_env = PresProcessGenerator(dataset_params, offline_env, dataset_params["random_seed_test"])
    

    num_episodes_test = dataset_params["test_size"]
    optimal_paths_test = []
    recommended_traces_test = []
    n_granted_test = 0
    cumulative_reward_test = []
    list_granted_test = []
    list_granted_percent_test = []
    list_len_ep_test = []
    case_id_tracker = []
    test_cases_with_impossible_actions = []

    start_time = time.time()

    total_visited_states = 0

    n_possible_transitions = 0
    n_impossible_transitions = 0

    #Q-learning implementation
    with tqdm(range(num_episodes_test), desc="Testing Progress", unit="episode") as pbar:
        for episode in pbar:
            done, granted = False, False
            impossible_action = False
            reward_episode, len_epis = 0, 0
            previous_trace = []

            event, simulation_state, event_scaled = online_env.reset_for_online(seed_to_add=episode)
            current_state = offline_env.build_state_with_lags(previous_trace, event_scaled)
            
            previous_trace.append(event_scaled)
            
            
            total_visited_states +=1

            while not done:
    
                    #select the best action according to policy network
                    with torch.no_grad():
                        action = algo.predict(current_state)[0]
                        action_name = [k for k, v in offline_env.activity_index.items() if v == action][0]
                
                    event, simulation_state, event_scaled, reward, granted, done, impossible_action = online_env.step_for_online(simulation_state, action=action_name)   
            

                    if not impossible_action:
                        n_possible_transitions +=1
                    else:
                        n_impossible_transitions += 1
                    
                    if event_scaled is not None: 
                        next_state = offline_env.build_state_with_lags(previous_trace, event_scaled)
                        previous_trace.append(event_scaled)
                        
                        total_visited_states +=1
                    else: 
                        next_state = None

                    current_state = next_state  
                    reward_episode += reward
                    len_epis += 1
                                    
            if granted:
                n_granted_test +=1

            test_cases_with_impossible_actions.append(impossible_action)
            case_id_tracker.append(episode)
            cumulative_reward_test.append(reward_episode)
            list_len_ep_test.append(len_epis)
            list_granted_test.append(granted)
            list_granted_percent_test.append(100*(n_granted_test/(episode+1)))
            recommended_traces_test.append(online_env.current_trace)
            optimal_paths_test.append(online_env.current_path)

            pbar.set_postfix(success=f"{100 * (n_granted_test / (episode + 1)):.2f}%", len_episode=f"{np.mean(list_len_ep_test):.1f}, impossible action taken: {sum(test_cases_with_impossible_actions)}")
            
    end_time = time.time()
    runtime = end_time - start_time 
    print(f"Online evaluation completed in {runtime:.2f} seconds.")

    optimal_paths_test = optimal_paths_test

    test_results = { "method": args.method,
                    "dataset": args.dataset,
                    "MDP_model":args.MDP_model,
                    "order": args.order,
                    'num_episodes': num_episodes_test,
                    "optimal_paths": optimal_paths_test,
                    "recommended_traces": recommended_traces_test,
                    "cumulative_reward": cumulative_reward_test,
                    "len_ep": list_len_ep_test,
                    "granted": list_granted_test,
                    "n_granted": list_granted_percent_test,
                    "total_visited_states": total_visited_states,
                    "test_cases_with_impossible_actions": test_cases_with_impossible_actions,
                    "test_case_ids": offline_env.test_case_ids,
                    "n_possible_transitions": n_possible_transitions,
                    "n_impossible_transitions" :n_impossible_transitions,
                    "args": vars(args)}
    return test_results