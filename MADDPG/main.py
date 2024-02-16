import collections
import numpy as np
from pettingzoo.mpe import simple_adversary_v3, simple_push_v3, simple_v3, simple_spread_v3
from MADDPG import MADDPG
from ma_replay_buffet import MultiAgenReplayBuffer
import wandb
import torch
import gym

def dict_to_list(a):
    groups = []
    for item in a:
        groups.append(list(item.values()))
    return  groups

INFERENCE = False
PRINT_INTERVAL = 5000
MAX_EPISODES = 50000
BATCH_SIZE = 1024
MAX_STEPS = 25
SEED = 10
total_steps = 0
score = -10
best_score = -100
score_history = []
WANDB = False
project_name = "MADDPG"
env_name = "simple_spread_v3"
env_class= simple_spread_v3
# env = simple_adversary_v3.env()
if WANDB:
    wandb.init(
        project=project_name,
        name=f"{env_name}_fastUp_{SEED}",
        group=env_name, 
        job_type=env_name,
        reinit=True
    )

import sys
sys.stdout = open('file_out.txt', 'w')
# print('Hello World!')
sys.stdout.close()
if INFERENCE:
    env = env_class.parallel_env(max_cycles=100, continuous_actions=True, render_mode="human")
else:
    env = env_class.parallel_env(continuous_actions=True)
    
# env = gym.wrappers.RecordEpisodeStatistics(env)
obs = env.reset(seed=SEED)
print(env)
print("num agents ", env.num_agents)
# print("observation space ", env.observation_spaces)
# print("action space ", env.action_spaces)
# print(obs)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

n_agents = env.num_agents
actor_dims = []
for i in range(n_agents):
    actor_dims.append(env.observation_space(env.agents[i]).shape[0])#s[list(env.observation_spaces.keys())[i]].shape[0])  

critic_dims = sum(actor_dims)
action_dim = env.action_space(env.agents[0]).shape[0]
maddpg = MADDPG(actor_dims, critic_dims, n_agents, action_dim,chkpt_dir="multi_agent/MADDPG/nets", scenario=f"/{env_name}", seed=SEED)
if INFERENCE:
    maddpg.load_checkpoint()
memory = MultiAgenReplayBuffer(critic_dims, actor_dims, action_dim,n_agents, batch_size=BATCH_SIZE, buffer_size=100000,seed = SEED)
# seed = 0
rewards_history = collections.deque(maxlen=100)
rewards_tot = collections.deque(maxlen=100)
for i in range(MAX_EPISODES):
    step = 0
    obs, info = env.reset(seed=SEED+i)
    obs=list(obs.values())
    done = [False] * n_agents
    rewards_ep_list = []

    
    # for agent in env.agent_iter():
    # observation, reward, termination, truncation, info = env.last()
    
    score = 0
    while  not any(done):#env.agents or
        # obss = [[ob] for ob in obs.values()]
        # obs = [ob for ob in obs.values()]
        # print(step, total_steps, i)
        
        actions = maddpg.choose_action(obs, INFERENCE, ep=i, max_ep=MAX_EPISODES, WANDB=WANDB)
        actions_dict = {agent:action.reshape(-1) for agent, action in zip(env.agents, actions)}
        data = env.step(actions_dict)
        data_processed = dict_to_list(data)
        obs_, rewards, terminations, truncations, info = data_processed
        done = (terminations or truncations)
        # if INFERENCE and done:
        #     env.render(render_mode="human")
        if step >= MAX_STEPS-1 and not INFERENCE:
            #print("MAX STEPS REACHED")
            done = [True] * n_agents
            # break
        if not INFERENCE:
            memory.store_transition(obs, actions, rewards, obs_, done)
        
        if (not INFERENCE) and total_steps % 50 == 0:
            maddpg.learn(memory)
        obs = obs_
        rewards_ep_list.append(rewards) 
        score += rewards[0]#+rewards["agent_1"]) #sum(rewards.values())
        step += 1
        total_steps += 1
    score_history.append(score)
    rewards_history.append(np.sum(rewards_ep_list, axis=0))
    rewards_tot.append(sum(rewards))
    # print('episode ', i, 'score %.1f' % score, 'memory length ', len(memory))
    avg_score = np.mean(rewards_tot)
    if WANDB and i % 100 == 0:    
        wandb.log({#'avg_score_adversary':np.mean(np.array(rewards_history)[:,0][0]),\
                # 'avg_score_agents':np.mean(np.array(rewards_history)[:,0][0]),\
                # 'avg_score_agent1':np.mean(np.array(rewards_history)[:,1][0]),\
                'total_rew':avg_score,'episode':i} )
           
    if i > 5000 and avg_score > best_score:
        print("episode: ", i, "avg: ", avg_score, "best: ", best_score)
        best_score = avg_score
        if not INFERENCE:
            print("Saving best model")
            maddpg.save_checkpoint()
    if i % PRINT_INTERVAL == 0 and i > 0:
        print('episode ', i, 'score %.1f' % score, 'avg score %.1f' % avg_score)


        # 

        