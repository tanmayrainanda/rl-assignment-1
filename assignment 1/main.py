import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
from my_package.sampler import sampler

class BanditArm:
    def __init__(self):
        self.total_reward = 0
        self.n_pulls = 0
        self.values = []
    
    def update(self, reward):
        self.n_pulls += 1
        self.total_reward += reward
        self.values.append(reward)

class ContextualBandit:
    def __init__(self, n_arms=4, n_contexts=3):
        self.n_arms = n_arms
        self.n_contexts = n_contexts
        self.arms = {i: {j: BanditArm() for j in range(n_arms)} for i in range(n_contexts)}
        
    def get_arm_value(self, context, arm):
        if self.arms[context][arm].n_pulls == 0:
            return 0
        return self.arms[context][arm].total_reward / self.arms[context][arm].n_pulls

class EpsilonGreedy(ContextualBandit):
    def __init__(self, epsilon, n_arms=4, n_contexts=3, decay_rate=0.995):
        super().__init__(n_arms, n_contexts)
        self.epsilon = epsilon
    
    def select_arm(self, context):
        if np.random.random() < self.epsilon:
            return np.random.randint(self.n_arms)
        values = [self.get_arm_value(context, arm) for arm in range(self.n_arms)]
        return np.argmax(values)

class UCB(ContextualBandit):
    def __init__(self, c, n_arms=4, n_contexts=3):
        super().__init__(n_arms, n_contexts)
        self.c = c
    
    def select_arm(self, context):
        for arm in range(self.n_arms):
            if self.arms[context][arm].n_pulls == 0:
                return arm
        
        total_pulls = sum(self.arms[context][arm].n_pulls for arm in range(self.n_arms))
        ucb_values = []
        
        for arm in range(self.n_arms):
            bonus = np.sqrt((2 * np.log(total_pulls)) / self.arms[context][arm].n_pulls)
            ucb_values.append(self.get_arm_value(context, arm) + self.c * bonus)
        
        return np.argmax(ucb_values)

class SoftMax(ContextualBandit):
    def __init__(self, temperature=1.0, n_arms=4, n_contexts=3):
        super().__init__(n_arms, n_contexts)
        self.temperature = temperature
    
    def select_arm(self, context):
        values = [self.get_arm_value(context, arm) for arm in range(self.n_arms)]
        exp_values = np.exp(np.array(values) / self.temperature)
        probabilities = exp_values / np.sum(exp_values)
        return np.random.choice(self.n_arms, p=probabilities)

class NewsRecommender:
    def __init__(self, roll_number):
        self.roll_number = roll_number
        self.reward_sampler = sampler(roll_number)
        self.user_classifier = DecisionTreeClassifier(random_state=42)
        self.label_encoder = LabelEncoder()
        
        # Initialize bandit models
        self.epsilon_greedy = EpsilonGreedy(epsilon=0.1)
        self.ucb = UCB(c=2)
        self.softmax = SoftMax(temperature=1.0)
        
        # Store rewards for analysis
        self.rewards_epsilon_greedy = {i: [] for i in range(3)}
        self.rewards_ucb = {i: [] for i in range(3)}
        self.rewards_softmax = {i: [] for i in range(3)}
        
        # Load and preprocess data
        self.preprocess_data()

    def preprocess_data(self):
        """Load and preprocess the data"""
        # Load user data
        self.train_users = pd.read_csv('train_users.csv')
        self.test_users = pd.read_csv('test_users.csv')
        self.news_articles = pd.read_csv('news_articles.csv')
        
        # Handle missing values if any
        self.train_users = self.train_users.fillna(0)
        self.test_users = self.test_users.fillna(0)
        
        # Encode categorical variables
        categorical_columns = self.train_users.select_dtypes(include=['object']).columns
        for col in categorical_columns:
            if col != 'label':
                self.label_encoder.fit(self.train_users[col])
                self.train_users[col] = self.label_encoder.transform(self.train_users[col])
                self.test_users[col] = self.label_encoder.transform(self.test_users[col])
        
        # Train user classifier
        X = self.train_users.drop('label', axis=1)
        y = self.train_users['label']
        self.user_classifier.fit(X, y)
        
        return self.train_users, self.test_users, self.news_articles

    def get_arm_index(self, context, arm):
        """
        Map context and arm to j value as per assignment specification
        context: 0-2 (User1, User2, User3)
        arm: 0-3 (Entertainment, Education, Tech, Crime)
        returns: j value (0-11)
        """
        return context * 4 + arm

    def train_bandits(self, n_iterations=10000):
        """Train all bandit models"""
        for _ in range(n_iterations):
            for context in range(3):  # For each user type
                # Train Epsilon-Greedy
                arm_eg = self.epsilon_greedy.select_arm(context)
                reward_eg = self.reward_sampler.sample(self.get_arm_index(context, arm_eg))
                self.epsilon_greedy.arms[context][arm_eg].update(reward_eg)
                self.rewards_epsilon_greedy[context].append(reward_eg)
                
                # Train UCB
                arm_ucb = self.ucb.select_arm(context)
                reward_ucb = self.reward_sampler.sample(self.get_arm_index(context, arm_ucb))
                self.ucb.arms[context][arm_ucb].update(reward_ucb)
                self.rewards_ucb[context].append(reward_ucb)
                
                # Train SoftMax
                arm_sm = self.softmax.select_arm(context)
                reward_sm = self.reward_sampler.sample(self.get_arm_index(context, arm_sm))
                self.softmax.arms[context][arm_sm].update(reward_sm)
                self.rewards_softmax[context].append(reward_sm)

    def compare_hyperparameters(self, epsilons=[0.1, 0.2, 0.3], c_values=[1, 2, 3], n_iterations=10000):
        """Compare performance with different hyperparameters"""
        results = {'epsilon_greedy': {}, 'ucb': {}}
        
        # Test different epsilon values
        for eps in epsilons:
            model = EpsilonGreedy(epsilon=eps)
            rewards = []
            for _ in range(n_iterations):
                for context in range(3):
                    arm = model.select_arm(context)
                    reward = self.reward_sampler.sample(self.get_arm_index(context, arm))
                    model.arms[context][arm].update(reward)
                    rewards.append(reward)
            results['epsilon_greedy'][eps] = np.mean(rewards)
        
        # Test different C values for UCB
        for c in c_values:
            model = UCB(c=c)
            rewards = []
            for _ in range(n_iterations):
                for context in range(3):
                    arm = model.select_arm(context)
                    reward = self.reward_sampler.sample(self.get_arm_index(context, arm))
                    model.arms[context][arm].update(reward)
                    rewards.append(reward)
            results['ucb'][c] = np.mean(rewards)
        
        self.plot_hyperparameter_comparison(results, epsilons, c_values)
        return results

    def plot_hyperparameter_comparison(self, results, epsilons, c_values):
        """Plot hyperparameter comparison results"""
        plt.figure(figsize=(15, 5))
        
        # Plot epsilon-greedy results
        plt.subplot(1, 2, 1)
        plt.plot(epsilons, [results['epsilon_greedy'][eps] for eps in epsilons], 'bo-')
        plt.title('Epsilon-Greedy Performance')
        plt.xlabel('Epsilon Value')
        plt.ylabel('Average Reward')
        plt.grid(True)
        
        # Plot UCB results
        plt.subplot(1, 2, 2)
        plt.plot(c_values, [results['ucb'][c] for c in c_values], 'ro-')
        plt.title('UCB Performance')
        plt.xlabel('C Value')
        plt.ylabel('Average Reward')
        plt.grid(True)
        
        plt.tight_layout()
        plt.show()

    def plot_rewards_by_context(self, window_size=100):
        """Plot average rewards over time for each context"""
        contexts = ['User1', 'User2', 'User3']
        plt.figure(figsize=(15, 12))
        
        for context_idx in range(3):
            plt.subplot(3, 1, context_idx + 1)
            
            # Calculate moving averages
            ma_eg = pd.Series(self.rewards_epsilon_greedy[context_idx]).rolling(window=window_size).mean()
            ma_ucb = pd.Series(self.rewards_ucb[context_idx]).rolling(window=window_size).mean()
            ma_sm = pd.Series(self.rewards_softmax[context_idx]).rolling(window=window_size).mean()
            
            plt.plot(ma_eg, label='Epsilon-Greedy')
            plt.plot(ma_ucb, label='UCB')
            plt.plot(ma_sm, label='SoftMax')
            
            plt.title(f'Average Reward Over Time - {contexts[context_idx]}')
            plt.xlabel('Iterations')
            plt.ylabel('Average Reward')
            plt.legend()
            plt.grid(True)
        
        plt.tight_layout()
        plt.show()

    def plot_detailed_performance(self, window_size=100):
        """Creates detailed plots for each user/context showing performance of all strategies"""
        contexts = ['user1', 'user2', 'user3']  # Updated to lowercase
        strategies = ['Epsilon-Greedy', 'UCB', 'SoftMax']
        
        fig, axes = plt.subplots(3, 3, figsize=(20, 15))
        plt.subplots_adjust(hspace=0.4, wspace=0.3)
        
        for context_idx in range(3):
            # Plot 1: Moving Average of Rewards
            ma_eg = pd.Series(self.rewards_epsilon_greedy[context_idx]).rolling(window=window_size).mean()
            ma_ucb = pd.Series(self.rewards_ucb[context_idx]).rolling(window=window_size).mean()
            ma_sm = pd.Series(self.rewards_softmax[context_idx]).rolling(window=window_size).mean()
            
            axes[context_idx, 0].plot(ma_eg, label='Epsilon-Greedy', color='blue')
            axes[context_idx, 0].plot(ma_ucb, label='UCB', color='green')
            axes[context_idx, 0].plot(ma_sm, label='SoftMax', color='red')
            axes[context_idx, 0].set_title(f'{contexts[context_idx]}: Moving Average Reward')
            axes[context_idx, 0].set_xlabel('Iterations')
            axes[context_idx, 0].set_ylabel('Average Reward')
            axes[context_idx, 0].legend()
            axes[context_idx, 0].grid(True)
            
            # Plot 2: Cumulative Rewards
            cum_eg = np.cumsum(self.rewards_epsilon_greedy[context_idx])
            cum_ucb = np.cumsum(self.rewards_ucb[context_idx])
            cum_sm = np.cumsum(self.rewards_softmax[context_idx])
            
            axes[context_idx, 1].plot(cum_eg, label='Epsilon-Greedy', color='blue')
            axes[context_idx, 1].plot(cum_ucb, label='UCB', color='green')
            axes[context_idx, 1].plot(cum_sm, label='SoftMax', color='red')
            axes[context_idx, 1].set_title(f'{contexts[context_idx]}: Cumulative Reward')
            axes[context_idx, 1].set_xlabel('Iterations')
            axes[context_idx, 1].set_ylabel('Cumulative Reward')
            axes[context_idx, 1].legend()
            axes[context_idx, 1].grid(True)
            
            # Plot 3: Reward Distribution (Box Plot)
            reward_data = [
                self.rewards_epsilon_greedy[context_idx],
                self.rewards_ucb[context_idx],
                self.rewards_softmax[context_idx]
            ]
            
            axes[context_idx, 2].boxplot(reward_data, labels=strategies)
            axes[context_idx, 2].set_title(f'{contexts[context_idx]}: Reward Distribution')
            axes[context_idx, 2].set_ylabel('Reward')
            axes[context_idx, 2].grid(True)
        
        plt.suptitle('Detailed Performance Analysis by User/Context', fontsize=16, y=1.02)
        plt.show()
        
        # Print statistical summary
        print("\nStatistical Summary:")
        print("-" * 50)
        for context_idx in range(3):
            print(f"\n{contexts[context_idx]}:")
            for strategy, rewards in [
                ('Epsilon-Greedy', self.rewards_epsilon_greedy[context_idx]),
                ('UCB', self.rewards_ucb[context_idx]),
                ('SoftMax', self.rewards_softmax[context_idx])
            ]:
                print(f"\n{strategy}:")
                print(f"  Mean Reward: {np.mean(rewards):.4f}")
                print(f"  Std Dev: {np.std(rewards):.4f}")
                print(f"  Max Reward: {np.max(rewards):.4f}")
                print(f"  Min Reward: {np.min(rewards):.4f}")

    def classify_user(self, user_features):
        """Classify a new user into user1/user2/user3"""
        user_class = self.user_classifier.predict([user_features])[0].lower()  # Convert to lowercase
        context_map = {'user1': 0, 'user2': 1, 'user3': 2}  # Updated to lowercase
        return context_map[user_class]

    def recommend_article(self, user_features):
        """Recommend article category and sample an article"""
        # Classify user
        context = self.classify_user(user_features)
        
        # Get recommendations from each model
        arm_eg = self.epsilon_greedy.select_arm(context)
        arm_ucb = self.ucb.select_arm(context)
        arm_sm = self.softmax.select_arm(context)
        
        # Map arms to categories
        category_map = {0: 'Entertainment', 1: 'Education', 2: 'Tech', 3: 'Crime'}
        recommendations = {
            'Epsilon-Greedy': category_map[arm_eg],
            'UCB': category_map[arm_ucb],
            'SoftMax': category_map[arm_sm]
        }
        
        # Use epsilon-greedy recommendation
        selected_category = recommendations['Epsilon-Greedy']
        
        # Check if the category exists in the dataset
        available_categories = self.news_articles['category'].unique()
        if selected_category not in available_categories:
            print(f"\nWarning: Category '{selected_category}' not found in dataset.")
            print(f"Available categories: {available_categories}")
            # Try to find a similar category (case-insensitive)
            selected_category_lower = selected_category.lower()
            for category in available_categories:
                if category.lower() == selected_category_lower:
                    selected_category = category
                    print(f"Using matching category: {category}")
                    break
        
        # Get articles from the selected category
        category_articles = self.news_articles[
            self.news_articles['category'].str.lower() == selected_category.lower()
        ]
        
        if len(category_articles) == 0:
            print(f"\nNo articles found in category: {selected_category}")
            recommended_article = None
        else:
            recommended_article = category_articles.sample(n=1).iloc[0]
        
        return recommendations, recommended_article

    def evaluate_models(self):
        """Evaluate performance of all models"""
        results = {
            'epsilon_greedy': {},
            'ucb': {},
            'softmax': {}
        }
        
        # Calculate average rewards per context
        for context in range(3):
            results['epsilon_greedy'][f'context_{context}'] = np.mean(self.rewards_epsilon_greedy[context])
            results['ucb'][f'context_{context}'] = np.mean(self.rewards_ucb[context])
            results['softmax'][f'context_{context}'] = np.mean(self.rewards_softmax[context])
        
        # Calculate cumulative regret
        optimal_reward = 1.0  # Assuming this is the maximum possible reward
        for model in ['epsilon_greedy', 'ucb', 'softmax']:
            rewards = getattr(self, f'rewards_{model.lower()}')
            cumulative_regret = {
                context: np.cumsum(optimal_reward - np.array(rewards[context]))
                for context in range(3)
            }
            results[model]['cumulative_regret'] = cumulative_regret
        
        # Calculate convergence time (iterations until stable reward)
        window = 100
        threshold = 0.01
        for model in ['epsilon_greedy', 'ucb', 'softmax']:
            rewards = getattr(self, f'rewards_{model.lower()}')
            for context in range(3):
                rolling_mean = pd.Series(rewards[context]).rolling(window=window).mean()
                rolling_std = pd.Series(rewards[context]).rolling(window=window).std()
                stable_point = np.where(rolling_std < threshold)[0]
                results[model][f'convergence_time_context_{context}'] = stable_point[0] if len(stable_point) > 0 else len(rewards[context])
        
        return results

    def generate_report(self):
        """Generate performance report for all models"""
        # Get evaluation results
        eval_results = self.evaluate_models()
        
        # Create rewards DataFrame
        rewards_data = {
            'Context': [],
            'Epsilon-Greedy': [],
            'UCB': [],
            'SoftMax': []
        }
        
        for context in range(3):
            rewards_data['Context'].append(f'Context {context}')
            rewards_data['Epsilon-Greedy'].append(eval_results['epsilon_greedy'][f'context_{context}'])
            rewards_data['UCB'].append(eval_results['ucb'][f'context_{context}'])
            rewards_data['SoftMax'].append(eval_results['softmax'][f'context_{context}'])
        
        rewards_df = pd.DataFrame(rewards_data)
        
        # Create convergence DataFrame
        convergence_data = {
            'Context': [],
            'Epsilon-Greedy': [],
            'UCB': [],
            'SoftMax': []
        }
        
        for context in range(3):
            convergence_data['Context'].append(f'Context {context}')
            convergence_data['Epsilon-Greedy'].append(
                eval_results['epsilon_greedy'][f'convergence_time_context_{context}']
            )
            convergence_data['UCB'].append(
                eval_results['ucb'][f'convergence_time_context_{context}']
            )
            convergence_data['SoftMax'].append(
                eval_results['softmax'][f'convergence_time_context_{context}']
            )
        
        convergence_df = pd.DataFrame(convergence_data)
        
        return rewards_df, convergence_df

def main():
    # Initialize recommender with your roll number
    recommender = NewsRecommender(roll_number=87)
    
    print("Starting news recommendation system training and evaluation...")
    
    # Train models
    print("\nTraining bandit models...")
    recommender.train_bandits(n_iterations=10000)
    
    # Generate plots
    print("\nGenerating performance plots...")
    print("1. Average rewards over time for each context")
    recommender.plot_rewards_by_context()
    
    print("\n2. Detailed performance analysis")
    recommender.plot_detailed_performance()
    
    print("\n3. Hyperparameter comparison")
    recommender.compare_hyperparameters(
        epsilons=[0.1, 0.2, 0.3],
        c_values=[1, 2, 3],
        n_iterations=10000
    )
    
    # Generate and display report
    rewards_df, convergence_df = recommender.generate_report()
    print("\nModel Performance Summary:")
    print("\nAverage Rewards per Context:")
    print(rewards_df)
    print("\nConvergence Time (iterations) per Context:")
    print(convergence_df)
    
    # Calculate classification accuracy
    y_true = recommender.test_users['label']
    y_pred = recommender.user_classifier.predict(recommender.test_users.drop('label', axis=1))
    accuracy = (y_true == y_pred).mean()
    print(f"\nUser Classification Accuracy: {accuracy:.2%}")
    
    # Test recommendation system
    print("\nTesting recommendation system...")
    test_user = recommender.test_users.iloc[0].drop('label')
    recommendations, article = recommender.recommend_article(test_user)
    
    print("\nRecommendations for test user:")
    for model, category in recommendations.items():
        print(f"{model}: {category}")
    print(f"\nSelected article: {article['headline']}")

if __name__ == "__main__":
    main()