"""
TD n-Step Q-Learning

Updates are based on a sequence of n steps into the future rather than just the immediate next step.

This means the algorithm waits until n steps are taken before updating the Q-value for the state-action pair (s, a) using the sequence of rewards observed and the maximum Q-value n steps later (s_n).

The Q-value update is based on the cumulative reward over n steps (sum of rewards received), plus the discounted maximum Q-value at the n-th step. This is known as the n-step return.
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

# n-step Q-learning
def n_step_q_learning(Q, N, episodes=10000):
    deltas = []
    for it in range(episodes):
        if it % 1000 == 0:
            print("Episode:", it)
        
        s = (2, 0)  # Start state
        grid.set_state(s)
        
        states_actions_rewards = []  # List to store state, action, and reward tuples
        for _ in range(N):
            a, _ = max_dict(Q[s])
            a = random_action(a)  # Choose an action
            r = grid.move(a)  # Take the action
            s2 = grid.current_state()  # Observe new state
            states_actions_rewards.append((s, a, r))
            s = s2
            if grid.game_over():
                break

        # Update Q-values using the experiences from the episode
        G = 0
        states_actions_rewards.reverse()
        for i, (s, a, r) in enumerate(states_actions_rewards):
            G = r + GAMMA * G  # Calculate return
            if i < N - 1:  # Skip the last N-1 steps
                continue
            old_qsa = Q[s][a]
            Q[s][a] += ALPHA * (G - Q[s][a])
            biggest_change = max(deltas, default=0, key=abs)
            deltas.append(np.abs(old_qsa - Q[s][a]))

    return deltas

deltas = n_step_q_learning(Q, N)

# Plot deltas to show learning progress
plt.plot(deltas)
plt.show()

# Determine the policy from Q*
policy = {}
V = {}
for s in grid.actions.keys():
    a, max_q = max_dict(Q[s])
    policy[s] = a
    V[s] = max_q

# Display the final policy and value function
print("final policy:")
print_policy(policy, grid)
print("final values:")
print_values(V, grid)
