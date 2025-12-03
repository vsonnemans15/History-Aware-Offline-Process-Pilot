# History-Aware Offline Process Pilot

## Summary of the Paper
Prescriptive process monitoring studies techniques that leverage event logs to recommend actions at runtime that improve business process outcomes. 
A promising approach for next-activity recommendations is reinforcement learning (RL).
Two challenges limit its broader adoption for business process optimization.
Firstly, as real-world trials are often infeasible, decisions must be learned from event logs. 
However, conventional RL methods trained on historical data often fail during deployment, as they tend to overestimate the value of unobserved actions and recommend them.
Secondly, conventional RL methods can recommend actions that violate the process control flow because they assume a memoryless decision-making process.
To address these limitations, this paper proposes a framework, referred to as History-Aware Offline _Process Pilot_, for learning optimal history-dependent recommendations from event logs, without any real-life or simulated trials.
The approach employs statistical testing to identify the window of historical context needed for decision-making and then trains an offline value-regularized RL algorithm on history-augmented observations. 
Our evaluation on synthetic and industrial event logs shows that the proposed method produces recommendations with better process outcomes and control-flow conformance than the benchmarks.

## Main Files Overview
#### `command.ls`
Contains all the python commands to replicate the results in the paper.

#### `src/`:
This directory contains all the Python scripts used for determining the Markov order of the decision-making process, training an offline CQL agent, and evaluating the policy.
- `MA_test.py`  
Applies the Markov Assumption statistical test by Shi et al. (2020) for `k = 1, ..., K`. 
- `CQL_offline.py`
Converts event data into k-step, history-augmented RL traces, trains an offline CQL agent to learn a history-dependent policy, and evaluates the learned policy using Fitted Q-Evaluation.
- `Evaluation.py`
Deploys the learned policy in the online setting of the synthetic SimBank process. 
- `MDP_functions.py`
Contains core functions for defining the Markov Decision Process, including state and terminal state definitions, reward computation, and removal of external actions from the agent's action space.

#### `SimBank/`:
This directory contains the SimBank implementation provided by De Moor et al.(2025).

#### `Final_plots/`:
This directory contains plots of CQL agent training losses recorded throughout the training iterations.

