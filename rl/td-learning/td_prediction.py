"""
TD Prediction

Temporal Difference (TD) Prediction is a reinforcement learning method used to estimate the value function of a policy by updating the value estimates based on the observed rewards and the estimated value of the next state, instead of waiting until the end of an episode as in Monte Carlo methods.

The provided code implements TD(0) Prediction, which is a specific instance of TD Prediction where the value of each state is updated based on the observed immediate reward and the estimated value of the next state. 

It iteratively updates the value function until convergence, using a specified step size parameter (ALPHA) and discount factor (GAMMA). The process involves repeatedly playing episodes using a given policy, updating the value function after each step, until the values converge to their optimal estimates.
"""

# https://github.com/omerbsezer/Reinforcement_learning_tutorial_with_demo/blob/master/td_prediction.ipynb
# the temporal difference 0 method to find the optimal policy
# only policy evaluation, not optimization

import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('/Users/williamphan/Desktop/model-a-day/')
from rl.envs.grid_world_game import standard_grid, negative_grid, print_values, print_policy

SMALL_ENOUGH = 1e-3
GAMMA = 0.9
ALL_POSSIBLE_ACTIONS = ('U', 'D', 'L', 'R')
ALPHA = 0.1


def random_action(a, eps=0.1):
    """
    Selects a random action with probability epsilon (eps), otherwise selects the given action 'a'.
    Parameters:
        a (str): The action to be selected with probability 1 - eps.
        eps (float): The probability of selecting a random action. Defaults to 0.1.
        
    Returns:
        str: The selected action.
    """
    p = np.random.random()
    if p < (1 - eps):
        return a
    else:
        return np.random.choice(ALL_POSSIBLE_ACTIONS)

def play_game(grid, policy):
    """
    Plays a game (episode) following the given policy and returns the sequence of states and rewards encountered.
    Parameters:
        grid (Grid): The grid world environment.
        policy (dict): The policy to be followed.
        
    Returns:
        list of tuple: A list of tuples representing the sequence of states and rewards encountered during the game.
    """
    s = (2, 0)
    grid.set_state(s)
    states_and_rewards = [(s, 0)]  # list of tuples of (state, reward)
    while not grid.game_over():
        a = policy[s]
        a = random_action(a)
        r = grid.move(a)
        s = grid.current_state()
        # After taking the action, grid.current_state() retrieves the next state, and (s, r) is 
        # appended to states_and_rewards, representing the tuple of next state and reward encountered.
        states_and_rewards.append((s, r))
    return states_and_rewards

grid = standard_grid()

# print rewards
print("rewards:")
print_values(grid.rewards, grid)

# state -> action
policy = {
  (2, 0): 'U',
  (1, 0): 'U',
  (0, 0): 'R',
  (0, 1): 'R',
  (0, 2): 'R',
  (1, 2): 'R',
  (2, 1): 'R',
  (2, 2): 'R',
  (2, 3): 'U',
}
# initial policy
print("initial policy:")
print_policy(policy, grid)

# initialize V(s) and returns
V = {}
states = grid.all_states()
for s in states:
  V[s] = 0
  
# initial value for all states in grid
print_values(V, grid)

# repeat until convergence
for it in range(1000):
  # generate an episode using pi
  states_and_rewards = play_game(grid, policy)
  # the first (s, r) tuple is the state we start in and 0
  # (since we don't get a reward) for simply starting the game
  # the last (s, r) tuple is the terminal state and the final reward
  # the value for the terminal state is by definition 0, so we don't
  # care about updating it.
  for t in range(len(states_and_rewards) - 1):
    s, _ = states_and_rewards[t]
    s2, r = states_and_rewards[t+1]
    # we will update V(s) AS we experience the episode
    # this line of code reflects the update step in the Temporal Difference (TD) learning method
    V[s] = V[s] + ALPHA*(r + GAMMA*V[s2] - V[s])
print("final values:")
print_values(V, grid)
print("final policy:")
print_policy(policy, grid)