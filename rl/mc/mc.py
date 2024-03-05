"""
Monte-Carlo

Monte Carlo methods in reinforcement learning use random sampling from episodes of interaction with an environment to estimate value functions and discover optimal policies.

The script conducts Monte Carlo policy evaluation in a grid world environment by simulating how an agent moves according to a predefined deterministic policy. 

The main point of the script, even when the policy remains unchanged before and after, is to perform policy evaluation. 

Steps:

1. Initialize the Grid: Set up the environment with predefined rewards and possible actions.

2. Define a Policy: Establish a deterministic policy that maps states to actions.

3. Simulate Episodes: For each iteration, simulate an episode by following the policy from a random starting state until a terminal state is reached, collecting states and rewards.

4. Calculate Returns: Work backwards from the end of each episode to compute the return for each state visited, applying a discount factor (GAMMA).

5. Update State Values: Average the returns for each state across episodes to estimate the state values under the policy.

6. Output Results: Print the estimated values of each state and the policy for review.
"""
# Adapted from: https://github.com/lazyprogrammer/machine_learning_examples/tree/master/rl
import numpy as np
import sys
sys.path.append('/Users/williamphan/Desktop/model-a-day/')
from rl.envs.grid_world_game import standard_grid, negative_grid, print_values, print_policy

# Constants for the Monte Carlo simulation
SMALL_ENOUGH = 1e-3  # Threshold for convergence
GAMMA = 0.9  # Discount factor for future rewards
ALL_POSSIBLE_ACTIONS = ('U', 'D', 'L', 'R')  # Possible actions

# Step 1: Initialize the Grid
def play_game(grid, policy):
    # Start the game in a random state from all possible actions
    start_states = list(grid.actions.keys())
    start_idx = np.random.choice(len(start_states))
    grid.set_state(start_states[start_idx])

    s = grid.current_state()
    states_and_rewards = [(s, 0)]  # Initial state with no reward

    # Step 3: Simulate Episodes
    while not grid.game_over():
        a = policy[s]  # Action from current policy
        r = grid.move(a)  # Reward from taking action
        s = grid.current_state()  # Update to new state
        states_and_rewards.append((s, r))  # Append state and reward

    G = 0
    states_and_returns = []
    first = True
    # Step 4: Calculate Returns
    for s, r in reversed(states_and_rewards):
        if first:
            first = False
        else:
            states_and_returns.append((s, G))
        G = r + GAMMA * G  # Discounted return
    states_and_returns.reverse()  # Reverse the list to the order of states visited

    return states_and_returns

# Load the grid
grid = standard_grid()

# Display rewards
print("rewards:")
print_values(grid.rewards, grid)

# Step 2: Define a Policy
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

# Display initial policy
print("initial policy:")
print_policy(policy, grid)

# Initialize V(s) and returns
V = {}
returns = {}  # Stores returns for state-action pairs
states = grid.all_states()

# Step 5: Update State Values
for s in states:
    if s in grid.actions:  # If state has associated actions
        returns[s] = []
    else:  # Terminal or non-reachable state
        V[s] = 0

# Main loop: Monte Carlo simulation to evaluate the policy
for t in range(100):  # Number of episodes
    states_and_returns = play_game(grid, policy)  # Generate an episode
    seen_states = set()
    for s, G in states_and_returns:
        if s not in seen_states:  # First-visit MC check
            returns[s].append(G)
            V[s] = np.mean(returns[s])  # Update state value based on average return
            seen_states.add(s)

# Step 6: Output Results
print("final values:")
print_values(V, grid)
print("final policy:")
print_policy(policy, grid)

"""
Rewards:

Shows the rewards for reaching specific states.

---------------------------
 0.00| 0.00| 0.00| 1.00|
---------------------------
 0.00| 0.00| 0.00|-1.00|
---------------------------
 0.00| 0.00| 0.00| 0.00|


Initial Policy:

Dictates the actions to be taken in each state. For example, 'R' means to move right, 
'U' means to move up, etc. Blank spaces represent terminal states or blocks where 
no action can be taken.

---------------------------
  R  |  R  |  R  |     |
---------------------------
  U  |     |  R  |     |
---------------------------
  U  |  R  |  R  |  U  |


Final Values:

Represent the long-term expected return from each state when following the initial policy. 
Values are discounted by a factor (GAMMA) over time, indicating the present value of future rewards. 

Higher values closer to the positive reward and lower (increasingly negative) values as 
they approach the negative reward.

---------------------------
 0.81| 0.90| 1.00| 0.00|
---------------------------
 0.73| 0.00|-1.00| 0.00|
---------------------------
 0.66|-0.81|-0.90|-1.00|


Final Policy:

The "final policy" in the output remains the same as the "initial policy," indicating that 
this phase of the Monte Carlo simulation was focused solely on policy evaluation, not 
policy improvement. During policy evaluation, the goal is to understand the value of each 
state under the current policy—essentially, how good it is to be in each state if you follow 
the given policy.

---------------------------
  R  |  R  |  R  |     |
---------------------------
  U  |     |  R  |     |
---------------------------
  U  |  R  |  R  |  U  |
"""