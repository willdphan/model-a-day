"""
Policy Iteration
Created: 03-03-2024
"""

# Adapted from: https://github.com/lazyprogrammer/machine_learning_examples/tree/master/rl
import numpy as np
import sys
sys.path.append('/Users/williamphan/Desktop/model-a-day/')
from rl.envs.grid_world_game import standard_grid, negative_grid, print_values, print_policy

SMALL_ENOUGH = 1e-3
GAMMA = 0.9
ALL_POSSIBLE_ACTIONS = ('U', 'D', 'L', 'R')
# this grid gives you a reward of -0.1
# to find a shorter path to the goal, use negative grid
grid = negative_grid()
print("rewards:")
print_values(grid.rewards, grid)

# state -> action
# choose an action and update randomly 
policy = {}
for s in grid.actions.keys():
  policy[s] = np.random.choice(ALL_POSSIBLE_ACTIONS)
# initial policy
print("initial policy:")
print_policy(policy, grid)

# initialize V(s) - value function
V = {}
states = grid.all_states()
for s in states:
  # V[s] = 0
  if s in grid.actions:
    V[s] = np.random.random()
  else:
    # terminal state
    V[s] = 0

# initial value for all states in grid
print(V)
print_values(V, grid)
{(0, 1): 0.48749319742265496, (1, 2): 0.43718927627766524, (0, 0): 0.5684833222359104, (2, 3): 0.1956657169276732, (2, 0): 0.4047423989785204, (1, 3): 0, (1, 0): 0.37995839505003837, (2, 2): 0.5256129770323059, (0, 3): 0, (0, 2): 0.031259508017021376, (2, 1): 0.04153421356916698}

iteration=0
# repeat until convergence
# when policy does not change, it will finish
while True:
  iteration+=1
  print("values %d: " % iteration)
  print_values(V, grid)
  print("policy %d: " % iteration)
  print_policy(policy, grid)

  # policy evaluation step
  while True:
    biggest_change = 0
    for s in states:
      old_v = V[s]

      # V(s) only has value if it's not a terminal state
      if s in policy:
        a = policy[s]
        grid.set_state(s)
        r = grid.move(a) #reward
        V[s] = r + GAMMA * V[grid.current_state()]
        biggest_change = max(biggest_change, np.abs(old_v - V[s]))

    if biggest_change < SMALL_ENOUGH:
      break

  # policy improvement step
  is_policy_converged = True
  for s in states:
    if s in policy:
      old_a = policy[s]
      new_a = None
      best_value = float('-inf')
      # loop through all possible actions to find the best current action
      for a in ALL_POSSIBLE_ACTIONS:
        grid.set_state(s)
        r = grid.move(a)
        v = r + GAMMA * V[grid.current_state()]
        if v > best_value:
          best_value = v
          new_a = a
      policy[s] = new_a
      if new_a != old_a:
        is_policy_converged = False

  if is_policy_converged:
    break

print("final values:")
print_values(V, grid)
print("final policy:")
print_policy(policy, grid)