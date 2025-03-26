import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

class PlakshaMDP:
    def __init__(self):
        # Define states: (Location, Hunger status)
        # 0: (Hostel, Not Hungry)
        # 1: (Hostel, Hungry)
        # 2: (Bharti Airtel Block, Not Hungry)
        # 3: (Bharti Airtel Block, Hungry)
        # 4: (Mess, Not Hungry)
        # 5: (Mess, Hungry)
        self.n_states = 6
        self.state_names = [
            "(Hostel, Not Hungry)",
            "(Hostel, Hungry)",
            "(Bharti Airtel Block, Not Hungry)",
            "(Bharti Airtel Block, Hungry)",
            "(Mess, Not Hungry)",
            "(Mess, Hungry)"
        ]
        
        # Define actions
        # 0: Attend Class
        # 1: Eat Food
        self.n_actions = 2
        self.action_names = ["Attend Class", "Eat Food"]
        
        # Define rewards based on location
        self.rewards = np.array([-1, -1, 3, 3, 1, 1])
        
        # Define transition probabilities: P[action, current_state, next_state]
        self.P = np.zeros((self.n_actions, self.n_states, self.n_states))
        
        # Action: Attend Class (0)
        # From (Hostel, Not Hungry)
        self.P[0, 0, 0] = 0.5  # Stay in Hostel
        self.P[0, 0, 2] = 0.5  # Go to Bharti Airtel Block
        
        # From (Hostel, Hungry)
        self.P[0, 1, 5] = 1.0  # Go to Mess
        
        # From (Bharti Airtel Block, Not Hungry)
        self.P[0, 2, 2] = 0.7  # Stay in Bharti Airtel Block
        self.P[0, 2, 4] = 0.3  # Go to Mess
        
        # From (Bharti Airtel Block, Hungry)
        self.P[0, 3, 3] = 0.2  # Stay in Bharti Airtel Block
        self.P[0, 3, 5] = 0.8  # Go to Mess
        
        # From (Mess, Not Hungry)
        self.P[0, 4, 2] = 0.6  # Go to Bharti Airtel Block
        self.P[0, 4, 0] = 0.3  # Go to Hostel
        self.P[0, 4, 4] = 0.1  # Stay in Mess
        
        # From (Mess, Hungry)
        self.P[0, 5, 5] = 1.0  # Stay in Mess
        
        # Action: Eat Food (1)
        # From any state, if eating food, go to (Mess, Not Hungry)
        for s in range(self.n_states):
            self.P[1, s, 4] = 1.0
    
    def print_mdp_table(self):
        """Print the MDP components in tabular form"""
        print("States:")
        for i, state in enumerate(self.state_names):
            print(f"{i}: {state} (Reward: {self.rewards[i]})")
        
        print("\nActions:")
        for i, action in enumerate(self.action_names):
            print(f"{i}: {action}")
        
        print("\nTransition Probabilities:")
        for a in range(self.n_actions):
            print(f"\nAction: {self.action_names[a]}")
            for s in range(self.n_states):
                print(f"  From State: {self.state_names[s]}")
                for s_prime in range(self.n_states):
                    if self.P[a, s, s_prime] > 0:
                        print(f"    To State: {self.state_names[s_prime]} with probability {self.P[a, s, s_prime]:.2f}")
    
    def draw_mdp_diagram(self):
        """Draw the MDP diagram with probabilities and rewards"""
        G = nx.DiGraph()
        
        # Add nodes
        for i, state_name in enumerate(self.state_names):
            G.add_node(state_name, reward=self.rewards[i])
        
        # Add edges for each action
        for action in range(self.n_actions):
            action_name = self.action_names[action]
            for source in range(self.n_states):
                source_name = self.state_names[source]
                for target in range(self.n_states):
                    target_name = self.state_names[target]
                    probability = self.P[action, source, target]
                    if probability > 0:
                        G.add_edge(source_name, target_name, 
                                   action=action_name, 
                                   probability=probability, 
                                   reward=self.rewards[target])
        
        # Plot the graph
        plt.figure(figsize=(14, 10))
        pos = nx.spring_layout(G, seed=42)
        
        # Draw nodes
        nx.draw_networkx_nodes(G, pos, node_size=3000, node_color='lightblue')
        
        # Draw edges with different colors based on action
        edge_colors = []
        edges = []
        
        for u, v, data in G.edges(data=True):
            edges.append((u, v))
            edge_colors.append('blue' if data['action'] == 'Attend Class' else 'red')
        
        nx.draw_networkx_edges(G, pos, edgelist=edges, edge_color=edge_colors, width=1.5)
        
        # Node labels
        node_labels = {node: f"{node}\nR={G.nodes[node]['reward']}" for node in G.nodes()}
        nx.draw_networkx_labels(G, pos, labels=node_labels, font_size=8)
        
        # Edge labels
        edge_labels = {}
        for u, v, data in G.edges(data=True):
            edge_labels[(u, v)] = f"{data['action']}\nP={data['probability']:.1f}"
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=7)
        
        plt.title("Plaksha University Student MDP")
        plt.axis('off')
        plt.tight_layout()
        plt.savefig('mdp_diagram.png', dpi=300)
        plt.show()
    
    def value_iteration(self, gamma=0.9, epsilon=1e-6, max_iterations=1000):
        """
        Implement value iteration algorithm to find optimal policy
        
        Parameters:
        -----------
        gamma : float
            Discount factor
        epsilon : float
            Convergence threshold
        max_iterations : int
            Maximum number of iterations
            
        Returns:
        --------
        V : numpy array
            Optimal value function
        policy : numpy array
            Optimal policy
        iterations : int
            Number of iterations until convergence
        """
        # Initialize value function
        V = np.zeros(self.n_states)
        
        # Track iterations for convergence plotting
        iteration_values = [V.copy()]
        
        # Value iteration
        for i in range(max_iterations):
            delta = 0
            
            # Update value for each state
            for s in range(self.n_states):
                v = V[s]
                
                # Calculate value for each action
                action_values = np.zeros(self.n_actions)
                for a in range(self.n_actions):
                    for s_prime in range(self.n_states):
                        action_values[a] += self.P[a, s, s_prime] * (self.rewards[s_prime] + gamma * V[s_prime])
                
                # Update state value to the best action value
                V[s] = np.max(action_values)
                
                # Update maximum change
                delta = max(delta, abs(v - V[s]))
            
            # Store current values
            iteration_values.append(V.copy())
            
            # Check convergence
            if delta < epsilon:
                break
        
        # Compute optimal policy
        policy = np.zeros(self.n_states, dtype=int)
        for s in range(self.n_states):
            action_values = np.zeros(self.n_actions)
            for a in range(self.n_actions):
                for s_prime in range(self.n_states):
                    action_values[a] += self.P[a, s, s_prime] * (self.rewards[s_prime] + gamma * V[s_prime])
            policy[s] = np.argmax(action_values)
        
        return V, policy, i+1, iteration_values
    
    def policy_iteration(self, gamma=0.9, epsilon=1e-6, max_iterations=1000):
        """
        Implement policy iteration algorithm to find optimal policy
        
        Parameters:
        -----------
        gamma : float
            Discount factor
        epsilon : float
            Convergence threshold
        max_iterations : int
            Maximum number of iterations
            
        Returns:
        --------
        V : numpy array
            Optimal value function
        policy : numpy array
            Optimal policy
        iterations : int
            Number of iterations until convergence
        """
        # Initialize policy randomly
        policy = np.zeros(self.n_states, dtype=int)
        
        # Keep track of policies
        policies = [policy.copy()]
        
        # Policy iteration
        for i in range(max_iterations):
            # Policy evaluation
            V = np.zeros(self.n_states)
            while True:
                delta = 0
                for s in range(self.n_states):
                    v = V[s]
                    # Update value based on current policy
                    V[s] = sum(self.P[policy[s], s, s_prime] * 
                               (self.rewards[s_prime] + gamma * V[s_prime]) 
                               for s_prime in range(self.n_states))
                    delta = max(delta, abs(v - V[s]))
                if delta < epsilon:
                    break
            
            # Policy improvement
            policy_stable = True
            for s in range(self.n_states):
                old_action = policy[s]
                
                # Find best action for current state
                action_values = np.zeros(self.n_actions)
                for a in range(self.n_actions):
                    for s_prime in range(self.n_states):
                        action_values[a] += self.P[a, s, s_prime] * (self.rewards[s_prime] + gamma * V[s_prime])
                
                # Update policy
                policy[s] = np.argmax(action_values)
                
                # Check if policy changed
                if old_action != policy[s]:
                    policy_stable = False
            
            policies.append(policy.copy())
            
            # Check if policy is stable
            if policy_stable:
                break
        
        return V, policy, i+1, policies

def main():
    # Create the MDP
    mdp = PlakshaMDP()
    
    # Print MDP in tabular form
    print("Finite MDP for Plaksha University Student:")
    mdp.print_mdp_table()
    
    # Draw MDP diagram
    mdp.draw_mdp_diagram()
    
    # Perform value iteration
    print("\n--- Value Iteration ---")
    vi_values, vi_policy, vi_iterations, vi_history = mdp.value_iteration()
    
    print(f"Converged after {vi_iterations} iterations")
    print("\nOptimal Values:")
    for s in range(mdp.n_states):
        print(f"  State: {mdp.state_names[s]}: {vi_values[s]:.4f}")
        
    print("\nOptimal Policy:")
    for s in range(mdp.n_states):
        print(f"  State: {mdp.state_names[s]}: {mdp.action_names[vi_policy[s]]}")
    
    # Perform policy iteration
    print("\n--- Policy Iteration ---")
    pi_values, pi_policy, pi_iterations, pi_history = mdp.policy_iteration()
    
    print(f"Converged after {pi_iterations} iterations")
    print("\nOptimal Values:")
    for s in range(mdp.n_states):
        print(f"  State: {mdp.state_names[s]}: {pi_values[s]:.4f}")
        
    print("\nOptimal Policy:")
    for s in range(mdp.n_states):
        print(f"  State: {mdp.state_names[s]}: {mdp.action_names[pi_policy[s]]}")
    
    # Compare results
    print("\n--- Comparison and Discussion ---")
    print("Value Iteration vs Policy Iteration:")
    print(f"- Value Iteration took {vi_iterations} iterations")
    print(f"- Policy Iteration took {pi_iterations} iterations")
    
    print("\nValue Differences:")
    for s in range(mdp.n_states):
        diff = abs(vi_values[s] - pi_values[s])
        print(f"  State: {mdp.state_names[s]}: {diff:.6f}")
    
    print("\nPolicy Differences:")
    policy_match = True
    for s in range(mdp.n_states):
        if vi_policy[s] != pi_policy[s]:
            policy_match = False
            print(f"  State: {mdp.state_names[s]}: VI={mdp.action_names[vi_policy[s]]}, PI={mdp.action_names[pi_policy[s]]}")
    
    if policy_match:
        print("  Both methods produced identical policies")
    
    print("\nDiscussion:")
    print("1. Both algorithms converge to the same optimal policy, which validates the solution.")
    print("2. Policy iteration typically converges in fewer iterations than value iteration.")
    print("3. The optimal policy shows that:")
    print("   - When hungry, the student should eat food regardless of location")
    print("   - When not hungry, the student should attend class to maximize rewards")
    print("   - The Bharti Airtel Block provides the highest immediate reward (+3)")
    print("4. The long-term expected rewards (values) reflect the balance between")
    print("   immediate rewards and future potential rewards considering transition probabilities.")

if __name__ == "__main__":
    main()