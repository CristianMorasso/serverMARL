from pettingzoo.mpe import simple_adversary_v3, simple_push_v3, simple_v3, simple_spread_v3
from MADDPG import MADDPG

env_class= simple_spread_v3
env_name = "simple_spread_v3"
env = env_class.parallel_env(max_cycles = 25, render_mode="human", continuous_actions=True)


INFERENCE = True
PRINT_INTERVAL = 1000
MAX_EPISODES = 300000
BATCH_SIZE = 1024
MAX_STEPS = 25
total_steps = 0
score = -10
best_score = 0
score_history = []
# env = simple_adversary_v3.env()
# obs = env.reset(seed=110)
observations, infos = env.reset(seed=111)
# print(env)
# print("num agents ", env.num_agents)
# print("observation space ", env.observation_spaces)
# print("action space ", env.action_spaces)
# print(obs)

n_agents = env.num_agents
actor_dims = []
# for i in range(n_agents):
#     actor_dims.append(env.observation_spaces[list(env.observation_spaces.keys())[i]].shape[0])  

# critic_dims = sum(actor_dims)
# action_dim = env.action_spaces[list(env.action_spaces.keys())[i]].shape[0]
for i in range(n_agents):
    actor_dims.append(env.observation_space(env.agents[i]).shape[0])#s[list(env.observation_spaces.keys())[i]].shape[0])  

critic_dims = sum(actor_dims)
action_dim = env.action_space(env.agents[0]).shape[0]
maddpg = MADDPG(actor_dims, critic_dims, n_agents, action_dim, chkpt_dir="nets", scenario=f"/{env_name}")
maddpg.load_checkpoint()
# maddpg.prep_rollouts(device='cpu')
# observations, infos = env.reset(seed=13)
for i in range(10):
    observations, infos = env.reset(seed=51+i)
    while env.agents:
        observations = list(observations.values())
        # this is where you would insert your policy
        actions = maddpg.choose_action(observations, INFERENCE)
        actions = {agent:action.reshape(-1) for agent, action in zip(env.agents, actions)}

        observations, rewards, terminations, truncations, infos = env.step(actions)
env.close()
