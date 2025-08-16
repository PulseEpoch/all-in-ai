import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import random
from collections import Counter


class DecisionTree:
    def __init__(self, max_depth=None, min_samples_split=2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.tree = None

    def entropy(self, y):
        """计算熵
        熵是衡量数据不确定性的指标，熵越高，数据的不确定性越大
        """
        counts = Counter(y)
        total = len(y)
        entropy = 0
        for count in counts.values():
            p = count / total
            entropy -= p * np.log2(p)
        return entropy

    def information_gain(self, X, y, feature_idx, threshold):
        """计算信息增益
        信息增益 = 父节点的熵 - 子节点的加权熵
        """
        # 分割数据
        left_mask = X[:, feature_idx] <= threshold
        right_mask = X[:, feature_idx] > threshold

        if len(y[left_mask]) == 0 or len(y[right_mask]) == 0:
            return 0

        # 计算父节点的熵
        parent_entropy = self.entropy(y)

        # 计算子节点的加权熵
        left_entropy = self.entropy(y[left_mask])
        right_entropy = self.entropy(y[right_mask])
        weight_left = len(y[left_mask]) / len(y)
        weight_right = len(y[right_mask]) / len(y)
        child_entropy = weight_left * left_entropy + weight_right * right_entropy

        # 计算信息增益
        return parent_entropy - child_entropy

    def best_split(self, X, y):
        """找到最佳的分割特征和阈值
        遍历所有特征和可能的阈值，选择信息增益最大的组合
        """
        best_gain = 0
        best_feature = None
        best_threshold = None

        n_features = X.shape[1]
        for feature_idx in range(n_features):
            # 获取该特征的所有唯一值
            thresholds = np.unique(X[:, feature_idx])
            for threshold in thresholds:
                gain = self.information_gain(X, y, feature_idx, threshold)
                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature_idx
                    best_threshold = threshold

        return best_feature, best_threshold

    def build_tree(self, X, y, depth=0):
        """递归构建决策树
        """
        # 基本情况：所有样本属于同一类别或达到最大深度
        if len(np.unique(y)) == 1 or depth == self.max_depth or len(y) < self.min_samples_split:
            return Counter(y).most_common(1)[0][0]

        # 找到最佳分割点
        best_feature, best_threshold = self.best_split(X, y)

        if best_feature is None:
            return Counter(y).most_common(1)[0][0]

        # 分割数据
        left_mask = X[:, best_feature] <= best_threshold
        right_mask = X[:, best_feature] > best_threshold

        # 递归构建左右子树
        left_subtree = self.build_tree(X[left_mask], y[left_mask], depth + 1)
        right_subtree = self.build_tree(X[right_mask], y[right_mask], depth + 1)

        # 返回决策节点
        return {
            'feature': best_feature,
            'threshold': best_threshold,
            'left': left_subtree,
            'right': right_subtree
        }

    def fit(self, X, y):
        """训练决策树
        """
        self.tree = self.build_tree(X, y)

    def predict_sample(self, x, tree):
        """预测单个样本
        """
        if not isinstance(tree, dict):
            return tree

        feature = tree['feature']
        threshold = tree['threshold']

        if x[feature] <= threshold:
            return self.predict_sample(x, tree['left'])
        else:
            return self.predict_sample(x, tree['right'])

    def predict(self, X):
        """预测多个样本
        """
        return np.array([self.predict_sample(x, self.tree) for x in X])


class RandomForest:
    def __init__(self, n_trees=10, max_depth=None, min_samples_split=2, max_features='sqrt'):
        self.n_trees = n_trees  # 树的数量
        self.max_depth = max_depth  # 树的最大深度
        self.min_samples_split = min_samples_split  # 节点分裂的最小样本数
        self.max_features = max_features  # 每棵树使用的最大特征数
        self.trees = []  # 存储所有树

    def bootstrap_sample(self, X, y):
        """自助采样
        从原始数据中有放回地采样，创建新的训练集
        """
        n_samples = X.shape[0]
        indices = np.random.choice(n_samples, n_samples, replace=True)
        return X[indices], y[indices]

    def fit(self, X, y):
        """训练随机森林
        """
        self.trees = []
        for _ in range(self.n_trees):
            # 自助采样
            X_sample, y_sample = self.bootstrap_sample(X, y)

            # 确定每棵树使用的特征数
            n_features = X.shape[1]
            if self.max_features == 'sqrt':
                max_features = int(np.sqrt(n_features))
            elif self.max_features == 'log2':
                max_features = int(np.log2(n_features))
            elif isinstance(self.max_features, int):
                max_features = self.max_features
            else:
                max_features = n_features

            # 随机选择特征
            feature_indices = np.random.choice(n_features, max_features, replace=False)
            X_sample = X_sample[:, feature_indices]

            # 创建并训练决策树
            tree = DecisionTree(max_depth=self.max_depth, min_samples_split=self.min_samples_split)
            tree.fit(X_sample, y_sample)

            # 保存树和使用的特征索引
            self.trees.append((tree, feature_indices))

    def predict(self, X):
        """预测多个样本
        对每个样本，取所有树的预测结果的多数投票
        """
        tree_preds = []
        for tree, feature_indices in self.trees:
            # 使用对应的特征子集进行预测
            X_subset = X[:, feature_indices]
            preds = tree.predict(X_subset)
            tree_preds.append(preds)

        # 多数投票
        tree_preds = np.array(tree_preds)
        majority_vote = [Counter(tree_preds[:, i]).most_common(1)[0][0] for i in range(X.shape[0])]
        return np.array(majority_vote)


# 示例用法
if __name__ == '__main__':
    # 加载数据集
    iris = load_iris()
    X, y = iris.data, iris.target

    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 训练随机森林
    rf = RandomForest(n_trees=10, max_depth=5, min_samples_split=2)
    rf.fit(X_train, y_train)

    # 预测
    y_pred = rf.predict(X_test)

    # 评估准确率
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Accuracy: {accuracy:.4f}')