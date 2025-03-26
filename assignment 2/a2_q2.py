import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import matplotlib.colors as mcolors
import random

class GridWorld:
    def __init__(self, grid_size=9):
        self.grid_size = grid_size
        
        # State space: (row, col)
        self.states = [(i, j) for i in range(grid_size) for j in range(grid_size)]
        
        # Action space: 0=up, 1=right, 2=down, 3=left
        self.actions = [0, 1, 2, 3]
        self.action_symbols = ['↑', '→', '↓', '←']
        self.action_deltas = [(-1, 0), (0, 1), (1, 0), (0, -1)]
        
        # Define walls (from the image)
        self.walls = [
            # Vertical walls
            (0, 5), (1, 5), (2, 5), (3, 5),  # Vertical wall at column 5
            (6, 3), (7, 3), (8, 3),  # Vertical wall at column 3
            
            # Horizontal walls
            (3, 5), (3, 6), (3, 7), (3, 8),  # Horizontal wall at row 3
            (6, 1), (6, 2), (6, 3)   # Horizontal wall at row 6
        ]
        
        # Define portals
        self.portal_in = (7, 2)    # IN portal
        self.portal_out = (2, 6)   # OUT portal
        
        # Define start and goal
        self.start = (8, 0)        # Robot starting position
        self.goal = (0, 8)         # Goal position (star)
        
        # Define terminal states
        self.terminal_states = [self.goal]
        
        # Discount factor
        self.gamma = 0.99
        
    def is_valid_state(self, state):
        """Check if a state is valid (within grid and not a wall)"""
        row, col = state
        return (0 <= row < self.grid_size and 
                0 <= col < self.grid_size and 
                state not in self.walls)
    
    def get_next_state(self, state, action):
        """Get the next state after taking an action from the current state"""
        # Handle portal IN -> OUT teleportation
        if state == self.portal_in:
            return self.portal_out
        
        row, col = state
        delta_row, delta_col = self.action_deltas[action]
        new_row, new_col = row + delta_row, col + delta_col
        new_state = (new_row, new_col)
        
        # Check if the new state is valid
        if self.is_valid_state(new_state):
            return new_state
        else:
            # If invalid, stay in the current state
            return state
    
    def get_reward(self, state, action, next_state):
        """Get the reward for a state-action-next_state transition"""
        if next_state == self.goal:
            return 1.0
        else:
            return 0.0
    
    def get_transition_prob(self, state, action, next_state):
        """
        Get the transition probability P(next_state | state, action)
        This is a deterministic environment, so prob is either 0 or 1
        """
        predicted_next_state = self.get_next_state(state, action)
        return 1.0 if predicted_next_state == next_state else 0.0
    
    def get_all_valid_states(self):
        """Get all valid states (not walls)"""
        return [s for s in self.states if s not in self.walls]
    
    def get_possible_next_states(self, state, action):
        """Get all possible next states and their probabilities"""
        next_state = self.get_next_state(state, action)
        return [(next_state, 1.0)]  # Deterministic: only one possible outcome
    

class ValueIteration:
    def __init__(self, env, theta=0.0001):
        self.env = env
        self.theta = theta  # Convergence threshold
        self.values = {state: 0.0 for state in env.get_all_valid_states()}
        self.policy = {state: 0 for state in env.get_all_valid_states()}
    
    def run(self, max_iterations=1000):
        """Run value iteration algorithm"""
        iteration_history = []
        
        for i in range(max_iterations):
            delta = 0
            
            # Store current values for history
            iteration_history.append(self.values.copy())
            
            # Update each state's value
            for state in self.env.get_all_valid_states():
                if state in self.env.terminal_states:
                    continue
                
                old_value = self.values[state]
                
                # Calculate the value of each action
                action_values = []
                for action in self.env.actions:
                    action_value = 0
                    
                    for next_state, prob in self.env.get_possible_next_states(state, action):
                        reward = self.env.get_reward(state, action, next_state)
                        action_value += prob * (reward + self.env.gamma * self.values[next_state])
                    
                    action_values.append(action_value)
                
                # Update the state value to the maximum action value
                self.values[state] = max(action_values)
                
                # Update delta
                delta = max(delta, abs(old_value - self.values[state]))
            
            # Check for convergence
            if delta < self.theta:
                print(f"Value Iteration converged after {i+1} iterations")
                break
        
        # Derive the optimal policy from the optimal values
        self.derive_policy()
        
        return self.values, self.policy, iteration_history
    
    def derive_policy(self):
        """Derive the optimal policy from the current value function"""
        for state in self.env.get_all_valid_states():
            if state in self.env.terminal_states:
                continue
                
            # Calculate the value of each action
            action_values = []
            for action in self.env.actions:
                action_value = 0
                
                for next_state, prob in self.env.get_possible_next_states(state, action):
                    reward = self.env.get_reward(state, action, next_state)
                    action_value += prob * (reward + self.env.gamma * self.values[next_state])
                
                action_values.append(action_value)
            
            # Select the action with the highest value
            self.policy[state] = np.argmax(action_values)


class PolicyIteration:
    def __init__(self, env, theta=0.0001):
        self.env = env
        self.theta = theta  # Convergence threshold
        self.values = {state: 0.0 for state in env.get_all_valid_states()}
        self.policy = {state: random.choice(env.actions) for state in env.get_all_valid_states()}
    
    def run(self, max_iterations=1000):
        """Run policy iteration algorithm"""
        policy_stable = False
        iteration = 0
        
        policy_history = []
        
        while not policy_stable and iteration < max_iterations:
            # Policy evaluation
            self.policy_evaluation()
            
            # Store policy for history
            policy_history.append(self.policy.copy())
            
            # Policy improvement
            policy_stable = self.policy_improvement()
            
            iteration += 1
        
        print(f"Policy Iteration converged after {iteration} iterations")
        return self.values, self.policy, policy_history
    
    def policy_evaluation(self, max_iterations=100):
        """Evaluate the current policy"""
        for _ in range(max_iterations):
            delta = 0
            
            for state in self.env.get_all_valid_states():
                if state in self.env.terminal_states:
                    continue
                
                old_value = self.values[state]
                
                # Get the action from current policy
                action = self.policy[state]
                
                # Calculate the value based on the policy
                new_value = 0
                for next_state, prob in self.env.get_possible_next_states(state, action):
                    reward = self.env.get_reward(state, action, next_state)
                    new_value += prob * (reward + self.env.gamma * self.values[next_state])
                
                self.values[state] = new_value
                
                # Update delta
                delta = max(delta, abs(old_value - self.values[state]))
            
            # Check for convergence
            if delta < self.theta:
                break
    
    def policy_improvement(self):
        """Improve the policy based on the current value function"""
        policy_stable = True
        
        for state in self.env.get_all_valid_states():
            if state in self.env.terminal_states:
                continue
            
            old_action = self.policy[state]
            
            # Calculate the value of each action
            action_values = []
            for action in self.env.actions:
                action_value = 0
                
                for next_state, prob in self.env.get_possible_next_states(state, action):
                    reward = self.env.get_reward(state, action, next_state)
                    action_value += prob * (reward + self.env.gamma * self.values[next_state])
                
                action_values.append(action_value)
            
            # Select the action with the highest value
            self.policy[state] = np.argmax(action_values)
            
            # Check if policy has changed
            if old_action != self.policy[state]:
                policy_stable = False
        
        return policy_stable


class MonteCarloApproximation:
    """
    A faster alternative to full Monte Carlo that uses direct value updates
    rather than complete episode generation, but still maintains the spirit
    of Monte Carlo by using sampling.
    """
    def __init__(self, env):
        self.env = env
        self.values = {state: 0.0 for state in env.get_all_valid_states()}
        self.policy = {state: 0 for state in env.get_all_valid_states()}
        self.Q = {(state, action): 0.0 
                 for state in env.get_all_valid_states() 
                 for action in env.actions}
        
        # Learning parameters
        self.alpha = 0.1  # Learning rate
        self.gamma = env.gamma  # Discount factor
    
    def run(self, num_iterations=10000):
        """
        Run a faster Monte Carlo approximation algorithm
        
        Instead of generating full episodes, we'll do direct updates
        to Q-values for each state-action pair and derive the policy.
        """
        valid_states = [s for s in self.env.get_all_valid_states() 
                       if s not in self.env.terminal_states]
        
        print(f"Running Monte Carlo approximation for {num_iterations} iterations...")
        
        for i in range(num_iterations):
            # Sample a random state
            state = random.choice(valid_states)
            
            # For each action, update Q-value using direct sampling
            for action in self.env.actions:
                # Get next state and reward
                next_state = self.env.get_next_state(state, action)
                reward = self.env.get_reward(state, action, next_state)
                
                # If terminal state, future value is 0
                if next_state in self.env.terminal_states:
                    future_value = 0
                else:
                    # Get best action from next state
                    next_action = max(self.env.actions, 
                                     key=lambda a: self.Q[(next_state, a)])
                    future_value = self.Q[(next_state, next_action)]
                
                # Update Q-value using Q-learning update rule
                target = reward + self.gamma * future_value
                self.Q[(state, action)] += self.alpha * (target - self.Q[(state, action)])
            
            # Periodically update the policy
            if i % 100 == 0 or i == num_iterations - 1:
                self._update_policy()
                print(f"Monte Carlo iteration {i}/{num_iterations}")
        
        # Final policy update
        self._update_policy()
        
        # Calculate state values from the policy
        for state in self.env.get_all_valid_states():
            if state in self.env.terminal_states:
                self.values[state] = 0
            else:
                self.values[state] = self.Q[(state, self.policy[state])]
        
        return self.values, self.policy
    
    def _update_policy(self):
        """Update policy to be greedy with respect to current Q-values"""
        for state in self.env.get_all_valid_states():
            if state in self.env.terminal_states:
                continue
            
            # Find action with highest Q-value
            self.policy[state] = max(self.env.actions, 
                                    key=lambda a: self.Q[(state, a)])


def visualize_policy(env, policy, title="Optimal Policy"):
    """Visualize the policy using a quiver plot"""
    # Create a grid for the visualization
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Add grid lines
    for i in range(env.grid_size + 1):
        ax.axhline(i, color='black', lw=0.5)
        ax.axvline(i, color='black', lw=0.5)
    
    # Plot walls
    for wall in env.walls:
        row, col = wall
        ax.add_patch(Rectangle((col, env.grid_size - 1 - row), 1, 1, 
                              facecolor='saddlebrown', edgecolor='black', alpha=0.7))
    
    # Plot portals
    in_row, in_col = env.portal_in
    ax.add_patch(Rectangle((in_col, env.grid_size - 1 - in_row), 1, 1, 
                          facecolor='lightblue', edgecolor='black', alpha=0.5))
    ax.text(in_col + 0.5, env.grid_size - 1 - in_row + 0.5, "IN", 
            ha='center', va='center', fontsize=12, fontweight='bold')
    
    out_row, out_col = env.portal_out
    ax.add_patch(Rectangle((out_col, env.grid_size - 1 - out_row), 1, 1, 
                          facecolor='lightblue', edgecolor='black', alpha=0.5))
    ax.text(out_col + 0.5, env.grid_size - 1 - out_row + 0.5, "OUT", 
            ha='center', va='center', fontsize=12, fontweight='bold')
    
    # Plot goal
    goal_row, goal_col = env.goal
    ax.add_patch(Rectangle((goal_col, env.grid_size - 1 - goal_row), 1, 1, 
                          facecolor='gold', edgecolor='black', alpha=0.5))
    ax.text(goal_col + 0.5, env.grid_size - 1 - goal_row + 0.5, "★", 
            ha='center', va='center', fontsize=24, color='darkblue')
    
    # Plot start
    start_row, start_col = env.start
    ax.add_patch(Rectangle((start_col, env.grid_size - 1 - start_row), 1, 1, 
                          facecolor='lightgray', edgecolor='black', alpha=0.3))
    ax.text(start_col + 0.5, env.grid_size - 1 - start_row + 0.5, "🤖", 
            ha='center', va='center', fontsize=16)
    
    # Prepare data for quiver plot
    X, Y = np.meshgrid(np.arange(0.5, env.grid_size, 1), np.arange(0.5, env.grid_size, 1))
    U = np.zeros_like(X)
    V = np.zeros_like(Y)
    
    # Set arrow directions based on policy
    for state in policy:
        row, col = state
        if state in env.terminal_states or state in env.walls:
            continue
        
        action = policy[state]
        
        # Convert to the quiver coordinate system (Y-axis is inverted)
        quiver_row = env.grid_size - 1 - row
        quiver_col = col
        
        # Set the arrow direction
        if action == 0:  # Up
            U[quiver_row, quiver_col] = 0
            V[quiver_row, quiver_col] = 1
        elif action == 1:  # Right
            U[quiver_row, quiver_col] = 1
            V[quiver_row, quiver_col] = 0
        elif action == 2:  # Down
            U[quiver_row, quiver_col] = 0
            V[quiver_row, quiver_col] = -1
        elif action == 3:  # Left
            U[quiver_row, quiver_col] = -1
            V[quiver_row, quiver_col] = 0
    
    # Draw the quiver plot
    ax.quiver(X, Y, U, V, scale=40, width=0.005, headwidth=4, headlength=4, color='blue')
    
    # Set plot limits and labels
    ax.set_xlim(0, env.grid_size)
    ax.set_ylim(0, env.grid_size)
    ax.set_aspect('equal')
    ax.set_title(title, fontsize=16)
    ax.set_xticks(np.arange(0.5, env.grid_size, 1))
    ax.set_yticks(np.arange(0.5, env.grid_size, 1))
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    plt.tight_layout()
    
    return fig, ax


def main():
    # Create the environment
    env = GridWorld()
    
    # Value Iteration
    print("Running Value Iteration...")
    vi = ValueIteration(env)
    vi_values, vi_policy, vi_history = vi.run()
    
    # Policy Iteration
    print("Running Policy Iteration...")
    pi = PolicyIteration(env)
    pi_values, pi_policy, pi_history = pi.run()
    
    # Monte Carlo Approximation
    print("Running Monte Carlo Approximation...")
    mc = MonteCarloApproximation(env)
    mc_values, mc_policy = mc.run(num_iterations=1000)
    
    # Visualize policies
    print("Visualizing policies...")
    
    # Value Iteration Policy
    vi_fig, _ = visualize_policy(env, vi_policy, "Value Iteration - Optimal Policy")
    vi_fig.savefig("value_iteration_policy.png", dpi=300, bbox_inches='tight')
    
    # Policy Iteration Policy
    pi_fig, _ = visualize_policy(env, pi_policy, "Policy Iteration - Optimal Policy")
    pi_fig.savefig("policy_iteration_policy.png", dpi=300, bbox_inches='tight')
    
    # Monte Carlo Policy
    mc_fig, _ = visualize_policy(env, mc_policy, "Monte Carlo Approximation - Optimal Policy")
    mc_fig.savefig("monte_carlo_policy.png", dpi=300, bbox_inches='tight')
    
    # Compare policies
    policy_diff_vi_pi = sum(1 for s in vi_policy if s not in env.walls and vi_policy[s] != pi_policy[s])
    policy_diff_vi_mc = sum(1 for s in vi_policy if s not in env.walls and vi_policy[s] != mc_policy[s])
    policy_diff_pi_mc = sum(1 for s in pi_policy if s not in env.walls and pi_policy[s] != mc_policy[s])
    
    print("\nPolicy Comparison:")
    print(f"Value Iteration vs Policy Iteration: {policy_diff_vi_pi} differences")
    print(f"Value Iteration vs Monte Carlo: {policy_diff_vi_mc} differences")
    print(f"Policy Iteration vs Monte Carlo: {policy_diff_pi_mc} differences")
    
    plt.show()

if __name__ == "__main__":
    main()