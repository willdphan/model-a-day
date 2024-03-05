"""
TD n-Step Sarsa

Updates the Q-value based on the sum of rewards over the next n steps, plus the discounted value of the action taken n steps ahead, according to the current policy. This allows it to consider a longer trajectory in the environment before making an update, potentially leading to a more informed update at each step.

Accumulates rewards over n steps and then updates the Q-values using these accumulated rewards. This approach can provide a more stable learning process since it incorporates more comprehensive feedback from the environment into each update.
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('/Users/williamphan/Desktop/model-a-day/')
from rl.envs.grid_world_game import standard_grid, negative_grid, print_values, print_policy
# Constants
GAMMA = 0.9
ALL_POSSIBLE_ACTIONS = ('U', 'D', 'L', 'R')
ALPHA = 0.1
N = 4  # Number of steps to look ahead

def max_dict(d):
    """Returns the argmax (key) and max (value) from a dictionary."""
    max_key = None
    max_val = float('-inf')
    for k, v in d.items():
        if v > max_val:
            max_val = v
            max_key = k
    return max_key, max_val

def random_action(a, eps=0.1):
    """Chooses an action using an epsilon-greedy approach."""
    p = np.random.random()
    if p < (1 - eps):
        return a
    else:
        return np.random.choice(ALL_POSSIBLE_ACTIONS)

grid = negative_grid(step_cost=-0.1)
print("rewards:")
print_values(grid.rewards, grid)

# Initialize Q(s, a)
Q = {}
states = grid.all_states()
for s in states:
    Q[s] = {}
    for a in ALL_POSSIBLE_ACTIONS:
        Q[s][a] = 0

# n-step SARSA
def n_step_sarsa(Q, N, episodes=10000):
    deltas = []
    for it in range(episodes):
        if it % 1000 == 0:
            print("Episode:", it)
        
        # Start of an episode
        s = (2, 0)
        grid.set_state(s)
        a = max_dict(Q[s])[0]
        a = random_action(a)

        # Initialize the first state and action
        states_actions_rewards = [(s, a, 0)]  # (state, action, reward)
        seen_states = set()
        seen_states.add(grid.current_state())
