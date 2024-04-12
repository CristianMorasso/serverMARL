import torch
from pettingzoo.mpe import simple_speaker_listener_v4
from agilerl.algorithms.maddpg import MADDPG
from agilerl.components.multi_agent_replay_buffer import MultiAgentReplayBuffer
import numpy as np
import pandas as pd 
import os
# if not os.path.isdir(f"{nets_out_dir}/{env_name}{params}"):
#     os.mkdir(f"{nets_out_dir}/{env_name}{params}")
import sys
sys.stdout = open('file_out.txt', 'w')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
env = simple_speaker_listener_v4.parallel_env(max_cycles=25, continuous_actions=True)
env.reset()

NET_CONFIG = {
      'arch': 'mlp',      # Network architecture
      'hidden_size': [32, 32]  # Network hidden size
  }
# Configure the multi-agent algo input arguments
try:
    state_dim = [env.observation_space(agent).n for agent in env.agents]
    one_hot = True
except Exception:
    state_dim = [env.observation_space(agent).shape for agent in env.agents]
    one_hot = False
try:
    action_dim = [env.action_space(agent).n for agent in env.agents]
    discrete_actions = True
    max_action = None
    min_action = None
except Exception:
    action_dim = [env.action_space(agent).shape[0] for agent in env.agents]
    discrete_actions = False
    max_action = [env.action_space(agent).high for agent in env.agents]
    min_action = [env.action_space(agent).low for agent in env.agents]

channels_last = False  # Swap image channels dimension from last to first [H, W, C] -> [C, H, W]
n_agents = env.num_agents
agent_ids = [agent_id for agent_id in env.agents]
field_names = ["state", "action", "reward", "next_state", "done"]
memory = MultiAgentReplayBuffer(memory_size=1_000_000,
                                field_names=field_names,
                                agent_ids=agent_ids,
                                device=device)

agent = MADDPG(state_dims=state_dim,
                action_dims=action_dim,
                one_hot=one_hot,
                n_agents=n_agents,
                agent_ids=agent_ids,
                max_action=max_action,
                min_action=min_action,
                discrete_actions=discrete_actions,
                device=device)

MAX_EPISODES = 1000
PRINT_INTERVAL = 100
best_score = -100
max_steps = 25 # For atari environments it is recommended to use a value of 500
epsilon = 1.0
eps_end = 0.1
eps_decay = 0.995
rewards_history = []
for ep in range(MAX_EPISODES):
    state, info  = env.reset() # Reset environment at start of episode
    agent_reward = {agent_id: 0 for agent_id in env.agents}
    if channels_last:
        state = {agent_id: np.moveaxis(np.expand_dims(s, 0), [3], [1]) for agent_id, s in state.items()}

    for _ in range(max_steps):
        agent_mask = info["agent_mask"] if "agent_mask" in info.keys() else None
        env_defined_actions = (
            info["env_defined_actions"]
            if "env_defined_actions" in info.keys()
            else None
        )

        # Get next action from agent
        cont_actions, discrete_action = agent.getAction(
            state, epsilon, agent_mask, env_defined_actions
        )
        if agent.discrete_actions:
            action = discrete_action
        else:
            action = cont_actions

        next_state, reward, termination, truncation, info = env.step(
            action
        )  # Act in environment
        done = termination 
        # Save experiences to replay buffer
        if channels_last:
            state = {agent_id: np.squeeze(s) for agent_id, s in state.items()}
            next_state = {agent_id: np.moveaxis(ns, [2], [0]) for agent_id, ns in next_state.items()}
        memory.save2memory(state, cont_actions, reward, next_state, done)

        for agent_id, r in reward.items():
            agent_reward[agent_id] += r

        # Learn according to learning frequency
        if (memory.counter % agent.learn_step == 0) and (len(
                memory) >= agent.batch_size):
            experiences = memory.sample(agent.batch_size) # Sample replay buffer
            agent.learn(experiences) # Learn according to agent's RL algorithm

        # Update the state
        if channels_last:
            next_state = {agent_id: np.expand_dims(ns,0) for agent_id, ns in next_state.items()}
        state = next_state

        # Stop episode if any agents have terminated
        if any(truncation.values()) or any(termination.values()):
            break

    # Save the total episode reward
    score = sum(agent_reward.values())
    agent.scores.append(score)

    # Update epsilon for exploration
    epsilon = max(eps_end, epsilon * eps_decay)

    
    avg_score = np.mean(agent.scores[-100:])
    rewards_history.append(avg_score)

    if ep > MAX_EPISODES/20 and avg_score > best_score:
        print("episode: ", ep, "avg: ", avg_score, "best: ", best_score)
        best_score = avg_score
    
        print("Saving best model")
        agent.saveCheckpoint(path="agile_maddpg_sl.ph")
    if ep % PRINT_INTERVAL == 0 and ep > 0:
        print('episode ', ep, 'score %.1f' % score, 'avg score %.1f' % avg_score)

reward_history_df = pd.DataFrame(rewards_history)
# reward_history_df.to_csv(f"{out_dir}/{env_name}{params}.csv")
reward_history_df.to_csv(f"agile_rl.csv")
print("-----END-----")
sys.stdout.close()