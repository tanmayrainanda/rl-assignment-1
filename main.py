import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
from my_package.sampler import sampler
import seaborn as sns

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

