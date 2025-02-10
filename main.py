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
