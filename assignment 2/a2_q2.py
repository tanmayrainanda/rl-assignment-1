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
            (5, 3), (6, 3), (7, 3),  # Vertical wall at column 3
            
            # Horizontal walls
            (3, 5), (3, 6), (3, 7), (3, 8),  # Horizontal wall at row 3
            (5, 1), (5, 2), (5, 3)   # Horizontal wall at row 6
        ]
        
        # Define portals
        self.portal_in = (6, 2)    # IN portal
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
    
    def run(self, num_iterations=1000):
        """
        Run a faster Monte Carlo approximation algorithm
        
        Instead of generating full episodes, we'll do direct updates
        to Q-values for each state-action pair and derive the policy.
        """
        valid_states = [s for s in self.env.get_all_valid_states() 
                       if s not in self.env.terminal_states]
        
        # Track value evolution at specific checkpoints
        checkpoint_iterations = []
        checkpoint_values = []
        
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
            
            # Periodically update the policy and save checkpoints
            if i % 100 == 0 or i == num_iterations - 1:
                self._update_policy()
                
                # Calculate current values for all states
                current_values = {state: self.Q[(state, self.policy[state])] 
                                 for state in self.env.get_all_valid_states() 
                                 if state not in self.env.terminal_states}
                
                # Store checkpoint
                checkpoint_iterations.append(i)
                checkpoint_values.append(current_values.copy())
                
                print(f"Monte Carlo iteration {i}/{num_iterations}")
        
        # Final policy update
        self._update_policy()
        
        # Calculate state values from the policy
        for state in self.env.get_all_valid_states():
            if state in self.env.terminal_states:
                self.values[state] = 0
            else:
                self.values[state] = self.Q[(state, self.policy[state])]
        
        return self.values, self.policy, (checkpoint_iterations, checkpoint_values)
    
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


def visualize_state_values(env, vi_values, pi_values, mc_values, title="State Value Comparison"):
    """Visualize the state values from different algorithms as heatmaps"""
    fig, axs = plt.subplots(1, 3, figsize=(18, 6))
    
    # Create 2D arrays for heatmaps
    vi_array = np.zeros((env.grid_size, env.grid_size))
    pi_array = np.zeros((env.grid_size, env.grid_size))
    mc_array = np.zeros((env.grid_size, env.grid_size))
    
    # Fill with NaN for walls
    for i in range(env.grid_size):
        for j in range(env.grid_size):
            if (i, j) in env.walls:
                vi_array[i, j] = np.nan
                pi_array[i, j] = np.nan
                mc_array[i, j] = np.nan
            else:
                vi_array[i, j] = vi_values.get((i, j), 0)
                pi_array[i, j] = pi_values.get((i, j), 0)
                mc_array[i, j] = mc_values.get((i, j), 0)
    
    # Create a common colormap range
    vmin = min(np.nanmin(vi_array), np.nanmin(pi_array), np.nanmin(mc_array))
    vmax = max(np.nanmax(vi_array), np.nanmax(pi_array), np.nanmax(mc_array))
    
    # Plot Value Iteration values
    im1 = axs[0].imshow(vi_array, cmap='viridis', vmin=vmin, vmax=vmax)
    axs[0].set_title('Value Iteration')
    
    # Plot Policy Iteration values
    im2 = axs[1].imshow(pi_array, cmap='viridis', vmin=vmin, vmax=vmax)
    axs[1].set_title('Policy Iteration')
    
    # Plot Monte Carlo values
    im3 = axs[2].imshow(mc_array, cmap='viridis', vmin=vmin, vmax=vmax)
    axs[2].set_title('Monte Carlo Approximation')
    
    # Add colorbars
    fig.colorbar(im1, ax=axs[0], label='State Value')
    fig.colorbar(im2, ax=axs[1], label='State Value')
    fig.colorbar(im3, ax=axs[2], label='State Value')
    
    # Add grid
    for ax in axs:
        for i in range(env.grid_size):
            ax.axhline(i - 0.5, color='white', linewidth=0.5)
            ax.axvline(i - 0.5, color='white', linewidth=0.5)
        
        # Mark special locations
        # Goal
        ax.plot(env.goal[1], env.goal[0], 'w*', markersize=10)
        # Start
        ax.plot(env.start[1], env.start[0], 'wo', markersize=8)
        # Portals
        ax.plot(env.portal_in[1], env.portal_in[0], 'ws', markersize=8)
        ax.plot(env.portal_out[1], env.portal_out[0], 'ws', markersize=8)
    
    plt.tight_layout()
    plt.savefig("state_values_comparison.png", dpi=300, bbox_inches='tight')
    
    return fig


def visualize_all_convergence(env, vi_history, pi_history, mc_iterations, mc_values_history):
    """
    Create comprehensive convergence visualizations for all three methods
    
    Args:
        env: The GridWorld environment
        vi_history: List of value function dictionaries for each VI iteration
        pi_history: List of policy dictionaries for each PI iteration
        mc_iterations: List of iteration numbers when MC values were recorded
        mc_values_history: List of value function dictionaries at MC checkpoints
    """
    # Create figure with 2x2 subplots
    fig, axs = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. Value Function Error/Change Plot (all methods)
    # Calculate maximum value change per iteration for each method
    
    # Value Iteration
    vi_diffs = []
    for i in range(1, len(vi_history)):
        max_diff = 0
        for state in vi_history[i]:
            diff = abs(vi_history[i][state] - vi_history[i-1][state])
            max_diff = max(max_diff, diff)
        vi_diffs.append(max_diff)
    
    # Policy Iteration - need to reconstruct value function history
    pi_values_history = []
    pi_diffs = []
    
    # Monte Carlo - calculate average change in values at checkpoints
    mc_diffs = []
    for i in range(1, len(mc_values_history)):
        avg_diff = 0
        count = 0
        for state in mc_values_history[i]:
            if state in mc_values_history[i-1]:
                diff = abs(mc_values_history[i][state] - mc_values_history[i-1][state])
                avg_diff += diff
                count += 1
        if count > 0:
            mc_diffs.append(avg_diff / count)
        else:
            mc_diffs.append(0)
    
    # Plot convergence of value functions (max delta)
    axs[0, 0].plot(range(1, len(vi_history)), vi_diffs, marker='.', label='Value Iteration')
    if len(mc_diffs) > 0:
        # Plot MC at the correct iteration numbers (not all iterations were saved)
        axs[0, 0].plot(mc_iterations[1:], mc_diffs, marker='o', label='Monte Carlo')
    axs[0, 0].set_xlabel('Iteration')
    axs[0, 0].set_ylabel('Value Function Change')
    axs[0, 0].set_title('Value Function Convergence')
    axs[0, 0].set_yscale('log')
    axs[0, 0].grid(True)
    axs[0, 0].legend()
    
    # 2. Policy Changes Plot
    pi_changes = []
    for i in range(1, len(pi_history)):
        changes = 0
        total = 0
        for state in pi_history[i]:
            if state not in env.walls and state not in env.terminal_states:
                total += 1
                if pi_history[i][state] != pi_history[i-1][state]:
                    changes += 1
        pi_changes.append(changes / max(1, total))  # As percentage of non-wall states
    
    axs[0, 1].plot(range(1, len(pi_history)), pi_changes, marker='.')
    axs[0, 1].set_xlabel('Iteration')
    axs[0, 1].set_ylabel('Policy Change Ratio')
    axs[0, 1].set_title('Policy Changes During Policy Iteration')
    axs[0, 1].grid(True)
    
    # 3. State Value Convergence for Key States (all methods)
    # Choose key states to track
    key_states = [
        env.start,       # Starting state
        (4, 4),          # Middle of grid
        (1, 7),          # Near goal
        (0, 7)           # One step from goal
    ]
    
    # Filter out any walls in key states
    key_states = [s for s in key_states if s not in env.walls]
    
    # For each key state, plot value over iterations for all methods
    for state in key_states:
        # Get value history for this state
        vi_values = [history.get(state, 0) for history in vi_history]
        
        # For Monte Carlo, we need to extract from the checkpoints
        mc_values = [history.get(state, 0) for history in mc_values_history]
        
        # Plot for this state
        axs[1, 0].plot(range(len(vi_history)), vi_values, 
                     label=f"VI State {state}")
        
        # Only plot MC if we have values
        if len(mc_values) > 0:
            axs[1, 0].plot(mc_iterations, mc_values, 
                         linestyle='--', marker='o',
                         label=f"MC State {state}")
    
    axs[1, 0].set_xlabel('Iteration')
    axs[1, 0].set_ylabel('State Value')
    axs[1, 0].set_title('Value Convergence for Key States')
    axs[1, 0].grid(True)
    axs[1, 0].legend()
    
    # 4. Bellman Error Plot (Value Iteration)
    vi_bellman_errors = []
    for i in range(len(vi_history)):
        values = vi_history[i]
        total_error = 0
        count = 0
        
        for state in values:
            if state in env.terminal_states or state in env.walls:
                continue
                
            # Calculate Bellman error
            max_action_value = float('-inf')
            for action in env.actions:
                action_value = 0
                for next_state, prob in env.get_possible_next_states(state, action):
                    reward = env.get_reward(state, action, next_state)
                    action_value += prob * (reward + env.gamma * values.get(next_state, 0))
                max_action_value = max(max_action_value, action_value)
            
            # Bellman error is the difference between value and the max action value
            error = abs(values[state] - max_action_value)
            total_error += error
            count += 1
        
        # Average error across all states
        if count > 0:
            vi_bellman_errors.append(total_error / count)
        else:
            vi_bellman_errors.append(0)
    
    axs[1, 1].plot(range(len(vi_history)), vi_bellman_errors, marker='.')
    axs[1, 1].set_xlabel('Iteration')
    axs[1, 1].set_ylabel('Average Bellman Error')
    axs[1, 1].set_title('Bellman Error During Value Iteration')
    axs[1, 1].set_yscale('log')
    axs[1, 1].grid(True)
    
    plt.tight_layout()
    plt.savefig("all_convergence_analysis.png", dpi=300, bbox_inches='tight')
    
    return fig


def visualize_convergence(vi_history, pi_history, title="Algorithm Convergence"):
    """Visualize the convergence of Value Iteration and Policy Iteration algorithms"""
    fig, axs = plt.subplots(2, 1, figsize=(12, 10))
    
    # For Value Iteration, plot the maximum absolute difference in values between iterations
    vi_diffs = []
    for i in range(1, len(vi_history)):
        max_diff = 0
        for state in vi_history[i]:
            diff = abs(vi_history[i][state] - vi_history[i-1][state])
            max_diff = max(max_diff, diff)
        vi_diffs.append(max_diff)
    
    # For Policy Iteration, count how many policy changes occur in each iteration
    pi_changes = []
    for i in range(1, len(pi_history)):
        changes = 0
        for state in pi_history[i]:
            if pi_history[i][state] != pi_history[i-1][state]:
                changes += 1
        pi_changes.append(changes)
    
    # Plot Value Iteration convergence
    axs[0].plot(range(1, len(vi_history)), vi_diffs, marker='.', label='Value Iteration')
    axs[0].set_xlabel('Iteration')
    axs[0].set_ylabel('Max Value Change')
    axs[0].set_title('Value Change per Iteration')
    axs[0].set_yscale('log')
    axs[0].grid(True)
    axs[0].legend()
    
    # Plot Policy Iteration policy changes
    axs[1].plot(range(1, len(pi_history)), pi_changes, marker='.')
    axs[1].set_xlabel('Iteration')
    axs[1].set_ylabel('Number of Policy Changes')
    axs[1].set_title('Policy Changes per Iteration (Policy Iteration)')
    axs[1].grid(True)
    
    plt.tight_layout()
    plt.savefig("convergence_comparison.png", dpi=300, bbox_inches='tight')
    
    return fig


def visualize_monte_carlo_convergence(env, checkpoint_iterations, checkpoint_values, title="Monte Carlo Convergence"):
    """Visualize how the Monte Carlo values evolve over time"""
    plt.figure(figsize=(10, 6))
    
    # Choose representative states to track
    key_states = [
        env.start,  # Start state
        (4, 4),     # Middle of the grid
        (1, 7),     # Near the goal
        (0, 7)      # One step from goal
    ]
    
    # Track values of these states over iterations
    for state in key_states:
        if state in env.walls:
            continue
            
        values_over_time = [values.get(state, 0) for values in checkpoint_values]
        plt.plot(checkpoint_iterations, values_over_time, 
                 marker='o', label=f"State {state}")
    
    plt.xlabel('Iteration')
    plt.ylabel('State Value')
    plt.title('Monte Carlo Value Convergence for Selected States')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("monte_carlo_convergence.png", dpi=300, bbox_inches='tight')
    
    return plt.gcf()


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
    mc_values, mc_policy, mc_checkpoint_data = mc.run(num_iterations=1000)
    mc_iterations, mc_values_checkpoints = mc_checkpoint_data
    
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
    
    # Visualize convergence for Value and Policy Iteration
    print("Visualizing convergence for Value and Policy Iteration...")
    convergence_fig = visualize_convergence(vi_history, pi_history)
    
    # Visualize Monte Carlo convergence separately
    print("Visualizing Monte Carlo convergence...")
    mc_convergence_fig = visualize_monte_carlo_convergence(env, mc_iterations, mc_values_checkpoints)
    
    # Visualize state values for all algorithms
    print("Visualizing state values for all algorithms...")
    values_fig = visualize_state_values(env, vi_values, pi_values, mc_values)
    
    # Create comprehensive convergence visualization for all methods
    print("Creating comprehensive convergence analysis...")
    all_convergence_fig = visualize_all_convergence(env, vi_history, pi_history, mc_iterations, mc_values_checkpoints)
    
    plt.show()


if __name__ == "__main__":
    main()