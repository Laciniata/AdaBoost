from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


def plot_decision_boundary(clf, X, y, axes=[-1.5, 2.45, -1, 1.5], alpha=0.5, contour=True):
    x1s = np.linspace(axes[0], axes[1], 100)
    x2s = np.linspace(axes[2], axes[3], 100)
    x1, x2 = np.meshgrid(x1s, x2s)
    X_new = np.c_[x1.ravel(), x2.ravel()]
    y_pred = clf.predict(X_new).reshape(x1.shape)
    custom_cmap = ListedColormap(['#fafab0', '#9898ff', '#a0faa0'])
    plt.contourf(x1, x2, y_pred, alpha=0.3, cmap=custom_cmap)
    if contour:
        custom_cmap2 = ListedColormap(['#7d7d58', '#4c4c7f', '#507d50'])
        plt.contour(x1, x2, y_pred, cmap=custom_cmap2, alpha=0.8)
    plt.plot(X[:, 0][y == -1], X[:, 1][y == -1], "yo", alpha=alpha)
    plt.plot(X[:, 0][y == 1], X[:, 1][y == 1], "bs", alpha=alpha)
    plt.axis(axes)
    plt.xlabel(r"$x_1$", fontsize=18)
    plt.ylabel(r"$x_2$", fontsize=18, rotation=0)


def data_init():
    frame = pd.read_csv("dataset/dataset.csv", index_col="编号")
    X = frame.loc[:, '属性1':'属性2'].to_numpy()
    y = frame.loc[:, '类别'].to_numpy()
    return X, y


if __name__ == '__main__':
    X, y = data_init()
    ada_clf = AdaBoostClassifier(DecisionTreeClassifier(max_depth=1), n_estimators=7, algorithm="SAMME.R",
                                 learning_rate=0.5)
    ada_clf.fit(X, y)
    y_pred = ada_clf.predict(X)
    print("查准率：", precision_score(y, y_pred))
    print("查全率：", recall_score(y, y_pred))
    print("F1 Score：", f1_score(y, y_pred))
    print("混淆矩阵：\n", confusion_matrix(y, y_pred))
    plot_decision_boundary(ada_clf, X, y)
    plt.show()
