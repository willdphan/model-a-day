"""
Monte Carlo Epsilon-Greedy
Created: 03-03-2024
---
The epsilon-greedy method balances exploration and exploitation in reinforcement learning by occasionally choosing a random action with probability ε, and otherwise selecting the action with the highest estimated value based on the current policy. 

In the provided code, the random_action function implements this method by choosing the given action with probability 1 - ε + ε/4 and selecting a random action with probability ε/4. This approach encourages the agent to explore different actions while still favoring those with higher estimated values according to the policy.

TLDR; Being "more greedy" means selecting actions that have higher estimated values according to the current action-value function. This adjustment is achieved through an epsilon-greedy strategy, where the policy is updated to choose the action with the highest estimated value most of the time, but with a small probability (epsilon), it explores other actions randomly to ensure continued exploration of the state space.
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('/Users/williamphan/Desktop/model-a-day/')
from rl.envs.grid_world_game import standard_grid, negative_grid, print_values, print_policy

SMALL_ENOUGH = 1e-3
GAMMA = 0.9
ALL_POSSIBLE_ACTIONS = ('U', 'D', 'L', 'R')

def random_action(a, eps=0.1):
  """
  Choose the given action with high probability and a random action with low probability.
  """
  # choose given a with probability 1 - eps + eps/4
  p = np.random.random()
  if p < (1 - eps):
    return a
  else:
    return np.random.choice(ALL_POSSIBLE_ACTIONS)
  
def max_dict(d):
  """
  Return the key and value corresponding to the maximum value in a dictionary.
  """
  # returns the argmax (key) and max (value) from a dictionary
  max_key = None
  max_val = float('-inf')
  for k, v in d.items():
    if v > max_val:
      max_val = v
      max_key = k
  return max_key, max_val

def play_game(grid, policy):
  """
  Simulate an episode using an epsilon-soft policy and return a list of states and corresponding returns.
  """
  # returns a list of states and corresponding returns
  # use an epsilon-soft policy
  s = (2, 0)
  grid.set_state(s)
  a = random_action(policy[s])

  # each triple is s(t), a(t), r(t)
  # but r(t) results from taking action a(t-1) from s(t-1) and landing in s(t)
  states_actions_rewards = [(s, a, 0)]
  while True:
    r = grid.move(a)
    s = grid.current_state()
    if grid.game_over():
      states_actions_rewards.append((s, None, r))
      break
    else:
      a = random_action(policy[s]) # the next state is stochastic
      states_actions_rewards.append((s, a, r))

  # calculate the returns by working backwards from the terminal state
  G = 0
  states_actions_returns = []
  first = True
  for s, a, r in reversed(states_actions_rewards):
    # the value of the terminal state is 0 by definition
    # we should ignore the first state we encounter
    # and ignore the last G, which is meaningless since it doesn't correspond to any move
    if first:
      first = False
    else:
      states_actions_returns.append((s, a, G))
    G = r + GAMMA*G
  states_actions_returns.reverse() # we want it to be in order of state visited
  return states_actions_returns

grid = negative_grid(step_cost=-0.1)
# print rewards
print("rewards:")
print_values(grid.rewards, grid)

# state -> action
# initialize a random policy
policy = {}
for s in grid.actions.keys():
  policy[s] = np.random.choice(ALL_POSSIBLE_ACTIONS)
  
# initial policy
print("initial policy:")
print_policy(policy, grid)

# initialize Q(s,a) and returns
Q = {}
returns = {} # dictionary of state -> list of returns we've received
states = grid.all_states()
for s in states:
  if s in grid.actions: # not a terminal state
    Q[s] = {}
    for a in ALL_POSSIBLE_ACTIONS:
      Q[s][a] = 0
      returns[(s,a)] = []
  else:
    # terminal state or state we can't otherwise get to
    pass
  
# initial Q values for all states in grid
print(Q)
{(0, 1): {'L': 0, 'D': 0, 'R': 0, 'U': 0}, (1, 2): {'L': 0, 'D': 0, 'R': 0, 'U': 0}, (0, 0): {'L': 0, 'D': 0, 'R': 0, 'U': 0}, (2, 3): {'L': 0, 'D': 0, 'R': 0, 'U': 0}, (2, 0): {'L': 0, 'D': 0, 'R': 0, 'U': 0}, (1, 0): {'L': 0, 'D': 0, 'R': 0, 'U': 0}, (2, 2): {'L': 0, 'D': 0, 'R': 0, 'U': 0}, (0, 2): {'L': 0, 'D': 0, 'R': 0, 'U': 0}, (2, 1): {'L': 0, 'D': 0, 'R': 0, 'U': 0}}

# repeat
deltas = []
for t in range(5000):
  # generate an episode using pi
  biggest_change = 0
  states_actions_returns = play_game(grid, policy)

  # calculate Q(s,a)
  seen_state_action_pairs = set()
  for s, a, G in states_actions_returns:
    # check if we have already seen s
    # called "first-visit" MC policy evaluation
    sa = (s, a)
    if sa not in seen_state_action_pairs:
      old_q = Q[s][a]
      returns[sa].append(G)
      Q[s][a] = np.mean(returns[sa])
      biggest_change = max(biggest_change, np.abs(old_q - Q[s][a]))
      seen_state_action_pairs.add(sa)
  deltas.append(biggest_change)

  # calculate new policy pi(s) = argmax[a]{ Q(s,a) }
  for s in policy.keys():
    a, _ = max_dict(Q[s])
    policy[s] = a
plt.plot(deltas)
plt.show()

# find the optimal state-value function
# V(s) = max[a]{ Q(s,a) }
V = {}
for s in policy.keys():
  V[s] = max_dict(Q[s])[1]

print("final values:")
print_values(V, grid)

print("final policy:")
print_policy(policy, grid)