"""
Monte Carlo On-Policy

The  script demonstrates an on-policy Monte Carlo simulation where an agent performs random walks in a two-dimensional space based on a heuristic policy. This policy biases the agent's movements towards returning to the origin, especially when the agent is considered "far away" (more than 3 units from the origin). 

These results show the effectiveness of the heuristic policy in guiding the agent back to the origin, or close enough to it, after a specified number of steps in a random walk. The "walk size" refers to the number of steps taken in each walk, and the "% of no transport" indicates the percentage of walks that ended within 4 units of the origin, suggesting that no additional transport would be needed to return to the origin.

Essentially, the policy used to generate behavior (actions taken by the agent) is the same as the policy being evaluated and improved. 

The idea is to learn from the actions you're currently taking.
"""

import random

# Define a policy function that decides the next move based on the current position
def heuristic_policy(x, y):
    """
    A simple heuristic policy guiding the agent to prefer moving towards the origin
    if it's considered far away. 'Far away' is defined as being more than 3 units from the origin.
    """
    # Define all possible actions the agent can take
    actions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
    
    # Modify the list of actions based on the agent's position to bias movement towards the origin
    if abs(x) + abs(y) > 3:  # Check if the agent is more than 3 units away from the origin
        # Remove actions that would take the agent further away from the origin
        if x < 0: actions.remove((-1, 0))  # If on the left, don't move further left
        elif x > 0: actions.remove((1, 0))  # If on the right, don't move further right
        if y < 0: actions.remove((0, -1))  # If below, don't move further down
        elif y > 0: actions.remove((0, 1))  # If above, don't move further up
    
    # Randomly select one of the remaining actions
    return random.choice(actions)

# Define the function that simulates the random walk according to the heuristic policy
def random_walk_on_policy(n):
    """Simulate a random walk of 'n' steps, where each step follows the heuristic policy."""
    x, y = 0, 0  # Start at the origin
    for _ in range(n):  # Repeat for 'n' steps
        dx, dy = heuristic_policy(x, y)  # Decide the next move based on the current position and policy
        x += dx  # Update the x-coordinate
        y += dy  # Update the y-coordinate
    return (x, y)  # Return the final position

# Set the number of random walks to simulate
number_of_walks = 20000

# Loop over a range of walk lengths to simulate walks of different lengths
for walk_length in range(1, 31):
    no_transport = 0  # Counter for how many walks end within 4 units of the origin
    
    # Simulate a number of walks for the current walk length
    for _ in range(number_of_walks):
        x, y = random_walk_on_policy(walk_length)  # Conduct a random walk
        distance = abs(x) + abs(y)  # Calculate the Manhattan distance from the origin
        
        # Check if the walk ends close enough to the origin to not need transport
        if distance <= 4:
            no_transport += 1  # Increment the counter if within 4 units
    
    # Calculate and print the percentage of walks that ended close to the origin
    print(f"Walk size = {walk_length}, % of no transport = {100 * no_transport / number_of_walks:.2f}%")

