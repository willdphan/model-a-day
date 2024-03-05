"""
Q-Learning, Off-Policy TD Control

Q-Learning is a type of reinforcement learning algorithm used to inform an agent on how to act optimally in a given environment by learning the value of actions in states without requiring a model of the environment. 

It aims to find the best action to take in a given state by learning the highest future rewards that action can yield, updating its knowledge as it explores the environment.

Script:

1. Initializing a table of Q-values to zero for all state-action pairs.

2. Exploring the environment with an epsilon-greedy strategy, where it sometimes chooses random actions to discover new rewards.

3. Updating the Q-values based on observed rewards and the maximum future rewards, using the formula: Q(s, a) = Q(s, a) + alpha * (reward + gamma * max(Q(s', a')) - Q(s, a)).

4. Repeating steps 2 and 3 until the Q-values converge to stable values, indicating the algorithm has learned.

5. Determining the optimal actions (policy) from the Q-values, guiding the agent to maximize rewards.
"""


# Adapted from: https://github.com/lazyprogrammer/machine_learning_examples/tree/master/rl
# https://github.com/omerbsezer/Reinforcement_learning_tutorial_with_demo/tree/master
# the Q-Learning method to find the optimal policy and value function
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('/Users/williamphan/Desktop/model-a-day/')
from rl.envs.grid_world_game import standard_grid, negative_grid, print_values, print_policy

SMALL_ENOUGH = 1e-3
GAMMA = 0.9
ALL_POSSIBLE_ACTIONS = ('U', 'D', 'L', 'R')
ALPHA = 0.1
def max_dict(d):
  # returns the argmax (key) and max (value) from a dictionary
  max_key = None
  max_val = float('-inf')
  for k, v in d.items():
    if v > max_val:
      max_val = v
      max_key = k
  return max_key, max_val
def random_action(a, eps=0.1):
  # epsilon-soft to ensure all states are visited
  p = np.random.random()
  if p < (1 - eps):
    return a
  else:
    return np.random.choice(ALL_POSSIBLE_ACTIONS)
grid = negative_grid(step_cost=-0.1)

# print rewards
print("rewards:")
print_values(grid.rewards, grid)

# no policy initialization, policy is derived from most recent Q like SARSA

# initialize Q(s,a)
Q = {}
states = grid.all_states()
for s in states:
  Q[s] = {}
  for a in ALL_POSSIBLE_ACTIONS:
    Q[s][a] = 0

# initial Q values for all states in grid
print(Q)
{(0, 1): {'L': 0, 'U': 0, 'D': 0, 'R': 0}, (1, 2): {'L': 0, 'U': 0, 'D': 0, 'R': 0}, (0, 0): {'L': 0, 'U': 0, 'D': 0, 'R': 0}, (2, 3): {'L': 0, 'U': 0, 'D': 0, 'R': 0}, (2, 0): {'L': 0, 'U': 0, 'D': 0, 'R': 0}, (1, 3): {'L': 0, 'U': 0, 'D': 0, 'R': 0}, (1, 0): {'L': 0, 'U': 0, 'D': 0, 'R': 0}, (2, 2): {'L': 0, 'U': 0, 'D': 0, 'R': 0}, (0, 3): {'L': 0, 'U': 0, 'D': 0, 'R': 0}, (0, 2): {'L': 0, 'U': 0, 'D': 0, 'R': 0}, (2, 1): {'L': 0, 'U': 0, 'D': 0, 'R': 0}}
update_counts = {}
update_counts_sa = {}
for s in states:
  update_counts_sa[s] = {}
  for a in ALL_POSSIBLE_ACTIONS:
    update_counts_sa[s][a] = 1.0
# repeat until convergence
t = 1.0
deltas = []
for it in range(10000):
  if it % 100 == 0:
    t += 1e-2
  if it % 2000 == 0:
    print("iteration:", it)

  # instead of 'generating' an epsiode, we will PLAY
  # an episode within this loop
  s = (2, 0) # start state
  grid.set_state(s)

  # the first (s, r) tuple is the state we start in and 0
  # (since we don't get a reward) for simply starting the game
  # the last (s, r) tuple is the terminal state and the final reward
  # the value for the terminal state is by definition 0, so we don't
  # care about updating it.
  a, _ = max_dict(Q[s])
  biggest_change = 0
  while not grid.game_over():
    a = random_action(a, eps=0.5/t) # epsilon-greedy
    # random action also works, but slower since you can bump into walls
    # a = np.random.choice(ALL_POSSIBLE_ACTIONS)
    r = grid.move(a)
    s2 = grid.current_state()

    # adaptive learning rate
    alpha = ALPHA / update_counts_sa[s][a]
    update_counts_sa[s][a] += 0.005

    # we will update Q(s,a) AS we experience the episode
    old_qsa = Q[s][a]
    # the difference between SARSA and Q-Learning is with Q-Learning
    # we will use this max[a']{ Q(s',a')} in our update
    # even if we do not end up taking this action in the next step
    a2, max_q_s2a2 = max_dict(Q[s2])
    Q[s][a] = Q[s][a] + alpha*(r + GAMMA*max_q_s2a2 - Q[s][a])
    biggest_change = max(biggest_change, np.abs(old_qsa - Q[s][a]))

    # we would like to know how often Q(s) has been updated too
    update_counts[s] = update_counts.get(s,0) + 1

    # next state becomes current state
    s = s2
    a = a2
   
  deltas.append(biggest_change)

plt.plot(deltas)
plt.show()

# determine the policy from Q*
# find V* from Q*
policy = {}
V = {}
for s in grid.actions.keys():
  a, max_q = max_dict(Q[s])
  policy[s] = a
  V[s] = max_q
print("update counts:")
total = np.sum(list(update_counts.values()))
for k, v in update_counts.items():
  update_counts[k] = float(v) / total
print_values(update_counts, grid)

print("final values:")
print_values(V, grid)
print("final policy:")
print_policy(policy, grid)
