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
    def __init__(self, epsilon, n_arms=4, n_contexts=3):
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
        
        self.epsilon_greedy = EpsilonGreedy(epsilon=0.1)
        self.ucb = UCB(c=2)
        self.softmax = SoftMax(temperature=1.0)
        
        self.rewards_epsilon_greedy = {i: [] for i in range(3)}
        self.rewards_ucb = {i: [] for i in range(3)}
        self.rewards_softmax = {i: [] for i in range(3)}
        
        # Create sample data
        self.news_articles = pd.DataFrame({
            'title': [f'Article {i}' for i in range(100)],
            'category': np.random.choice(['Entertainment', 'Education', 'Tech', 'Crime'], 100),
            'content': [f'Content {i}' for i in range(100)]
        })
        
        user_features = {
            'feature1': np.random.rand(100),
            'feature2': np.random.rand(100),
            'feature3': np.random.rand(100),
            'label': np.random.choice(['User1', 'User2', 'User3'], 100)
        }
        self.train_users = pd.DataFrame(user_features)
        self.test_users = pd.DataFrame({k: v[:20] for k, v in user_features.items()})
    
    def preprocess_data(self):
        categorical_columns = self.train_users.select_dtypes(include=['object']).columns
        for col in categorical_columns:
            if col != 'label':
                self.label_encoder.fit(self.train_users[col])
                self.train_users[col] = self.label_encoder.transform(self.train_users[col])
                self.test_users[col] = self.label_encoder.transform(self.test_users[col])
        
        X = self.train_users.drop('label', axis=1)
        y = self.train_users['label']
        self.user_classifier.fit(X, y)
        
        return self.train_users, self.test_users, self.news_articles
    
    def classify_user(self, user_features):
        user_class = self.user_classifier.predict([user_features])[0]
        # Map user class to context number
        context_map = {'User1': 0, 'User2': 1, 'User3': 2}
        return context_map[user_class]
    
    def get_arm_index(self, context, arm):
        return context * 4 + arm
    
    def train_bandits(self, n_iterations=10000):
        for _ in range(n_iterations):
            for context in range(3):
                arm_eg = self.epsilon_greedy.select_arm(context)
                reward_eg = self.reward_sampler.sample(self.get_arm_index(context, arm_eg))
                self.epsilon_greedy.arms[context][arm_eg].update(reward_eg)
                self.rewards_epsilon_greedy[context].append(reward_eg)
                
                arm_ucb = self.ucb.select_arm(context)
                reward_ucb = self.reward_sampler.sample(self.get_arm_index(context, arm_ucb))
                self.ucb.arms[context][arm_ucb].update(reward_ucb)
                self.rewards_ucb[context].append(reward_ucb)
                
                arm_sm = self.softmax.select_arm(context)
                reward_sm = self.reward_sampler.sample(self.get_arm_index(context, arm_sm))
                self.softmax.arms[context][arm_sm].update(reward_sm)
                self.rewards_softmax[context].append(reward_sm)
    
    def recommend_article(self, user_features):
        context = self.classify_user(user_features)
        arm = self.epsilon_greedy.select_arm(context)
        
        category_map = {0: 'Entertainment', 1: 'Education', 2: 'Tech', 3: 'Crime'}
        recommended_category = category_map[arm]
        
        category_articles = self.news_articles[self.news_articles['category'] == recommended_category]
        if len(category_articles) == 0:
            return recommended_category, None
        
        recommended_article = category_articles.sample(n=1).iloc[0]
        return recommended_category, recommended_article
    
    def evaluate_models(self):
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
        optimal_reward = 1.0
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

    def plot_rewards(self, window_size=100):
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        for context in range(3):
            ma_eg = pd.Series(self.rewards_epsilon_greedy[context]).rolling(window=window_size).mean()
            ma_ucb = pd.Series(self.rewards_ucb[context]).rolling(window=window_size).mean()
            ma_sm = pd.Series(self.rewards_softmax[context]).rolling(window=window_size).mean()
            
            axes[context].plot(ma_eg, label='Epsilon-Greedy')
            axes[context].plot(ma_ucb, label='UCB')
            axes[context].plot(ma_sm, label='SoftMax')
            axes[context].set_title(f'Context {context+1}')
            axes[context].set_xlabel('Iterations')
            axes[context].set_ylabel('Average Reward')
            axes[context].legend()
        
        plt.tight_layout()
        plt.show()

    def plot_regret(self):
        plt.figure(figsize=(12, 4))
        models = ['epsilon_greedy', 'ucb', 'softmax']
        colors = ['b', 'g', 'r']
        
        for context in range(3):
            plt.subplot(1, 3, context + 1)
            for model, color in zip(models, colors):
                regret = self.evaluate_models()[model]['cumulative_regret'][context]
                plt.plot(regret, label=model, color=color)
            plt.title(f'Context {context+1} Cumulative Regret')
            plt.xlabel('Iterations')
            plt.ylabel('Cumulative Regret')
            plt.legend()
        
        plt.tight_layout()
        plt.show()

    def compare_hyperparameters(self, epsilons=[0.1, 0.2, 0.3], c_values=[1, 2, 3], n_iterations=10000):
        results = {
            'epsilon_greedy': {},
            'ucb': {}
        }
        
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
        
        # Plot results
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        plt.plot(epsilons, [results['epsilon_greedy'][eps] for eps in epsilons], 'bo-')
        plt.title('Epsilon-Greedy Performance')
        plt.xlabel('Epsilon Value')
        plt.ylabel('Average Reward')
        
        plt.subplot(1, 2, 2)
        plt.plot(c_values, [results['ucb'][c] for c in c_values], 'ro-')
        plt.title('UCB Performance')
        plt.xlabel('C Value')
        plt.ylabel('Average Reward')
        
        plt.tight_layout()
        plt.show()
        
        return results

    def analyze_context_performance(self):
        """Analyze performance for each context separately"""
        contexts = ['User1', 'User2', 'User3']
        models = ['Epsilon-Greedy', 'UCB', 'SoftMax']
        
        plt.figure(figsize=(15, 5))
        
        for i, context in enumerate(range(3)):
            rewards = {
                'Epsilon-Greedy': self.rewards_epsilon_greedy[context],
                'UCB': self.rewards_ucb[context],
                'SoftMax': self.rewards_softmax[context]
            }
            
            plt.subplot(1, 3, i+1)
            plt.boxplot([rewards[model] for model in models], labels=models)
            plt.title(f'Rewards Distribution - {contexts[i]}')
            plt.ylabel('Reward')
            plt.xticks(rotation=45)
        
        plt.tight_layout()
        plt.show()
        
        # Statistical summary
        summary = {}
        for context in range(3):
            summary[contexts[context]] = {
                'Epsilon-Greedy': {
                    'mean': np.mean(self.rewards_epsilon_greedy[context]),
                    'std': np.std(self.rewards_epsilon_greedy[context]),
                    'max': np.max(self.rewards_epsilon_greedy[context])
                },
                'UCB': {
                    'mean': np.mean(self.rewards_ucb[context]),
                    'std': np.std(self.rewards_ucb[context]),
                    'max': np.max(self.rewards_ucb[context])
                },
                'SoftMax': {
                    'mean': np.mean(self.rewards_softmax[context]),
                    'std': np.std(self.rewards_softmax[context]),
                    'max': np.max(self.rewards_softmax[context])
                }
            }
        
        return pd.DataFrame(summary)

def main():
    recommender = NewsRecommender(roll_number=87)
    train_users, test_users, news_articles = recommender.preprocess_data()
    recommender.train_bandits(n_iterations=10000)
    # Plot performance metrics
    recommender.plot_rewards()
    recommender.plot_regret()
    
    # Generate and display report
    rewards_df, convergence_df = recommender.generate_report()
    print("\nAverage Rewards per Context:")
    print(rewards_df)
    print("\nConvergence Time (iterations) per Context:")
    print(convergence_df)
    
    user_features = test_users.iloc[0].drop('label')
    category, article = recommender.recommend_article(user_features)
    print(f"Recommended Category: {category}")
    if article is not None:
        print(f"Recommended Article: {article['title']}")

if __name__ == "__main__":
    main()