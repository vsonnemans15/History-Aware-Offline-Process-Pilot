# History-Aware Offline Process Pilot

Prescriptive process monitoring studies techniques that leverage event logs to recommend actions at runtime that improve business process outcomes. 
A promising approach for next-activity recommendations is reinforcement learning (RL).
Two challenges limit its broader adoption for business process optimization.
Firstly, as real-world trials are often infeasible, decisions must be learned from event logs. 
However, conventional RL methods trained on historical data often fail during deployment, as they tend to overestimate the value of unobserved actions and recommend them.
Secondly, conventional RL methods can recommend actions that violate the process control flow because they assume a memoryless decision-making process.
To address these limitations, this paper proposes a framework, referred to as History-Aware Offline _Process Pilot_, for learning optimal history-dependent recommendations from event logs, without any real-life or simulated trials.
The approach employs statistical testing to identify the window of historical context needed for decision-making and then trains an offline value-regularized RL algorithm on history-augmented observations. 
Our evaluation on synthetic and industrial event logs shows that the proposed method produces recommendations with better process outcomes and control-flow conformance than the benchmarks.
