"""
Monte Carlo (MC) Control with Exploring Starts is a method in reinforcement learning used to find an optimal policy by combining policy evaluation and policy improvement.

Exploring Starts method is a technique used in reinforcement learning to ensure exploration of all state-action pairs by starting episodes from random initial states and actions. 

In standard MC methods, the episodes typically start from the current state of the agent and follow the policy from that state onwards. This approach does not guarantee that all state-action pairs will be visited during the learning process.

In contrast, the Exploring Starts method ensures that each state-action pair has a nonzero probability of being selected as the starting point of an episode. This is achieved by randomly selecting state-action pairs to initiate episodes, thus promoting exploration and ensuring that all state-action pairs are eventually visited and evaluated.

Steps:

1. Initialization: It initializes the grid world, defines constants like small threshold value, discount factor, and possible actions.

2. Play Game Function: The play_game function generates an episode using the exploring starts method. It simulates an episode where the agent interacts with the environment according to a given policy. 

The exploring starts ensure that every state-action pair has a non-zero probability of being selected as the starting point of an episode.

3. Policy Initialization: It initializes a random policy where each state is mapped to a random action.

4. Q-Value and Returns Initialization: It initializes Q-values and returns for each state-action pair. The Q-values represent the expected return for taking a particular action in a particular state.

5. Policy Evaluation and Improvement Loop: It iteratively evaluates the policy by playing episodes and 
updating the Q-values based on the returns obtained. It also improves the policy by updating it to be 
greedy with respect to the current Q-values.

6. Convergence Monitoring: It tracks the change in Q-values over iterations to monitor convergence.

7. Plotting: It plots the change in Q-values over iterations to visualize the  onvergence process.

8. Final Policy and Values: It prints the final policy and the corresponding state values obtained after convergence.
"""

# Adapted from: https://github.com/lazyprogrammer/machine_learning_examples/tree/master/rl
# the Monte Carlo Exploring-Starts method to find the optimal policy
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('/Users/williamphan/Desktop/model-a-day/')
from rl.envs.grid_world_game import standard_grid, negative_grid, print_values, print_policy

SMALL_ENOUGH = 1e-3
GAMMA = 0.9
ALL_POSSIBLE_ACTIONS = ('U', 'D', 'L', 'R')
def play_game(grid, policy):
  # returns a list of states and corresponding returns
  # we have a deterministic policy
  # we would never end up at certain states, but we still want to measure their value
  # this is called the "exploring starts" method
  start_states = list(grid.actions.keys())
  start_idx = np.random.choice(len(start_states))
  grid.set_state(start_states[start_idx])

  s = grid.current_state()
  a = np.random.choice(ALL_POSSIBLE_ACTIONS) # first action is uniformly random

  # each triple s(t), a(t), r(t)
  # but r(t) results from taking action a(t-1) from s(t-1) and landing in s(t)
  states_actions_rewards = [(s, a, 0)]
  seen_states = set()
  seen_states.add(grid.current_state())
  num_steps = 0
  while True:
    r = grid.move(a)
    num_steps += 1
    s = grid.current_state()

    if s in seen_states:
      # we don't end up in an infinitely long episode
      # bumping into the wall repeatedly
      reward = -10. / num_steps
      states_actions_rewards.append((s, None, reward))
      break
    elif grid.game_over():
      states_actions_rewards.append((s, None, r))
      break
    else:
      a = policy[s]
      states_actions_rewards.append((s, a, r))
    seen_states.add(s)

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
def max_dict(d):
  # returns the argmax (key) and max (value) from a dictionary
  max_key = None
  max_val = float('-inf')
  for k, v in d.items():
    if v > max_val:
      max_val = v
      max_key = k
  return max_key, max_val
grid = negative_grid(step_cost=-0.9)
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
      Q[s][a] = 0 # needs to be initialized to something so we can argmax it
      returns[(s,a)] = []
  else:
    # terminal state or state we can't otherwise get to
    pass

# initial Q values for all states in grid
print(Q)
{(0, 1): {'U': 0, 'R': 0, 'L': 0, 'D': 0}, (1, 2): {'U': 0, 'R': 0, 'L': 0, 'D': 0}, (0, 0): {'U': 0, 'R': 0, 'L': 0, 'D': 0}, (2, 3): {'U': 0, 'R': 0, 'L': 0, 'D': 0}, (2, 0): {'U': 0, 'R': 0, 'L': 0, 'D': 0}, (1, 0): {'U': 0, 'R': 0, 'L': 0, 'D': 0}, (2, 2): {'U': 0, 'R': 0, 'L': 0, 'D': 0}, (0, 2): {'U': 0, 'R': 0, 'L': 0, 'D': 0}, (2, 1): {'U': 0, 'R': 0, 'L': 0, 'D': 0}}

# repeat until convergence
deltas = []
for t in range(2000):
  # generate an episode using pi
  biggest_change = 0
  states_actions_returns = play_game(grid, policy)
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

  # update policy
  for s in policy.keys():
    policy[s] = max_dict(Q[s])[0]
plt.plot(deltas)
plt.show()

print("final policy:")
print_policy(policy, grid)


# find V
V = {}
for s, Qs in Q.items():
  V[s] = max_dict(Q[s])[1]

print("final values:")
print_values(V, grid)

"""
Rewards:

---------------------------
-0.90|-0.90|-0.90| 1.00|
---------------------------
-0.90| 0.00|-0.90|-1.00|
---------------------------
-0.90|-0.90|-0.90|-0.90|

Initial Policy:

The initial policy is randomly initialized, as shown in the printed output. Each state is associated 
with a randomly chosen action.

---------------------------
  U  |  L  |  U  |     |
---------------------------
  L  |     |  D  |     |
---------------------------
  U  |  U  |  D  |  R  |


Final Policy:

After running the Monte Carlo method, the final policy is obtained. The printed output shows the optimal 
action to take in each state according to the learned policy.

---------------------------
  R  |  R  |  R  |     |
---------------------------
  U  |     |  U  |     |
---------------------------
  R  |  R  |  U  |  U  |


Final Values:

The final state values are printed, indicating the expected return or utility of being in each state. 
Higher values correspond to more desirable states.

---------------------------
-1.75|-0.95| 1.00| 0.00|
---------------------------
-2.70| 0.00|-0.76| 0.00|
---------------------------
-3.10|-2.85|-1.86|-1.00|
"""