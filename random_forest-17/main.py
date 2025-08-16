import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris, load_wine, load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import seaborn as sns
from random_forest import RandomForest


def plot_feature_importance(X, y, feature_names):
    """绘制特征重要性
    通过遍历每个特征，打乱该特征的值，观察准确率下降的程度来评估特征重要性
    """
    # 训练原始模型
    rf = RandomForest(n_trees=10, max_depth=5)
    rf.fit(X, y)
    baseline_accuracy = accuracy_score(y, rf.predict(X))

    # 评估每个特征的重要性
    importance = []
    for i in range(X.shape[1]):
        # 复制数据
        X_permuted = X.copy()
        # 打乱当前特征的值
        np.random.shuffle(X_permuted[:, i])
        # 预测
        y_pred = rf.predict(X_permuted)
        # 计算准确率下降
        accuracy = accuracy_score(y, y_pred)
        importance.append(baseline_accuracy - accuracy)

    # 绘制特征重要性
    plt.figure(figsize=(10, 6))
    plt.bar(feature_names, importance)
    plt.title('特征重要性')
    plt.xlabel('特征')
    plt.ylabel('准确率下降')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('feature_importance.png')
    plt.close()

    return importance


def plot_trees_vs_accuracy(X_train, X_test, y_train, y_test):
    """绘制树的数量与准确率的关系
    """
    n_trees_range = [1, 3, 5, 10, 20, 30, 50]
    train_accuracy = []
    test_accuracy = []

    for n_trees in n_trees_range:
        # 训练模型
        rf = RandomForest(n_trees=n_trees, max_depth=5)
        rf.fit(X_train, y_train)

        # 评估训练集准确率
        y_pred_train = rf.predict(X_train)
        train_accuracy.append(accuracy_score(y_train, y_pred_train))

        # 评估测试集准确率
        y_pred_test = rf.predict(X_test)
        test_accuracy.append(accuracy_score(y_test, y_pred_test))

    # 绘制图表
    plt.figure(figsize=(10, 6))
    plt.plot(n_trees_range, train_accuracy, 'o-', label='训练集准确率')
    plt.plot(n_trees_range, test_accuracy, 's-', label='测试集准确率')
    plt.title('树的数量与准确率的关系')
    plt.xlabel('树的数量')
    plt.ylabel('准确率')
    plt.legend()
    plt.grid(True)
    plt.savefig('trees_vs_accuracy.png')
    plt.close()


def visualize_confusion_matrix(y_test, y_pred, class_names):
    """可视化混淆矩阵
    """
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title('混淆矩阵')
    plt.xlabel('预测类别')
    plt.ylabel('真实类别')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png')
    plt.close()


def main():
    # 加载数据集（这里使用鸢尾花数据集作为示例）
    dataset = load_iris()
    X, y = dataset.data, dataset.target
    feature_names = dataset.feature_names
    class_names = dataset.target_names

    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 训练随机森林模型
    rf = RandomForest(n_trees=10, max_depth=5, min_samples_split=2)
    rf.fit(X_train, y_train)

    # 在测试集上进行预测
    y_pred = rf.predict(X_test)

    # 评估模型性能
    accuracy = accuracy_score(y_test, y_pred)
    print(f'准确率: {accuracy:.4f}')
    print('分类报告:')
    print(classification_report(y_test, y_pred, target_names=class_names))

    # 可视化混淆矩阵
    visualize_confusion_matrix(y_test, y_pred, class_names)

    # 绘制特征重要性
    feature_importance = plot_feature_importance(X_train, y_train, feature_names)
    print('特征重要性:')
    for name, importance in zip(feature_names, feature_importance):
        print(f'{name}: {importance:.4f}')

    # 绘制树的数量与准确率的关系
    plot_trees_vs_accuracy(X_train, X_test, y_train, y_test)

    print('所有可视化结果已保存到当前目录')


if __name__ == '__main__':
    main()