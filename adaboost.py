import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


def data_init():
    """
    数据初始化。

    Returns:
        data (pd.DataFrame): 数据集
    """
    data = pd.read_csv("dataset/dataset.csv")
    data.drop('编号', axis=1, inplace=True)
    return data


def divide_point(data: pd.DataFrame):
    """
    两两求中点。
    Args:
        data (pd.DataFrame): 数据集

    Returns:
        dvd_pts (dict): 字典，key:类型，value:中点列表
    """
    dvd_pts = {}
    for c in data.columns:
        if c == '类别':
            break
        sorted_c = data[c].sort_values()
        mids = [(sorted_c.iloc[i] + sorted_c.iloc[i + 1]) / 2 for i in range(len(sorted_c) - 1)]
        dvd_pts[c] = mids
    return dvd_pts


def best_precision_with_weight(data: pd.DataFrame, dvd_attribute: str, dvd_pt: np.float64, weight: list):
    """
    已知属性和划分点，计算精度最高的两侧分类及其精度。
    Args:
        data (pd.DataFrame): 数据集
        dvd_attribute (str): 划分属性
        dvd_pt (np.float64): 划分点
        weight (list): 样本权重（分布）

    Returns:
        precision (float): 最高精度
        left_category (np.int64): 小于划分点的样本的分类
        right_category (np.int64): 大于划分点的样本的分类
    """
    weight = np.array(weight)
    weight = (weight / weight.sum()).flatten().tolist()  # 归一化weight
    categories = data['类别'].unique()
    # 对大于、小于的求两类各自权重之和
    data['权重'] = weight
    less_than_judgment = data[dvd_attribute] < dvd_pt
    weight_sums = {'lt': {}, 'gt': {}}
    for category in categories:
        mask_lt = (data[less_than_judgment]['类别'] == category)
        weight_sum_lt = data.where(mask_lt).dropna()['权重'].sum()
        weight_sums['lt'][category] = weight_sum_lt
        mask_gt = (data[~less_than_judgment]['类别'] == category)
        weight_sum_gt = data.where(mask_gt).dropna()['权重'].sum()
        weight_sums['gt'][category] = weight_sum_gt
    weight_sums_frame = pd.DataFrame(weight_sums)
    precision = sum([weight_sums_frame[c].max() for c in weight_sums_frame.columns])  # 大于、小于均取权重和最大的为分类结果
    left_category = weight_sums_frame['lt'].idxmax()
    right_category = weight_sums_frame['gt'].idxmax()
    data.drop("权重", axis=1, inplace=True)  # 最后删除权重列
    return precision, left_category, right_category


def decision_stump_fit(data: pd.DataFrame, weight: np.ndarray = None):
    """
    训练决策树桩，标准为最小化加权分类误差。决策树桩数据结构：(分类属性，划分点，左类，右类)。

    Args:
        data (pd.DataFrame): 数据集
        weight (np.ndarray): 样本权重（分布）

    Returns:
        决策树桩：
            best_attribute (str): 划分属性
            best_dvd_pt (np.float64): 划分点
            less_than_category (np.int64): 小于划分点的样本的分类
            greater_than_category (np.int64): 大于划分点的样本的分类
    """
    if weight is None:  # 无权重，则所有权重相等
        weight = np.ones(data.shape[0])
        weight = (weight / weight.sum()).tolist()
    dvd_pts = divide_point(data)
    precisions = {}  # 最佳精度列表
    classify_results = {}  # 对应最佳精度的大于、小于两侧分类
    for attribute in dvd_pts.keys():
        precisions[attribute] = []
        classify_results[attribute] = []
        for dvd_pt in dvd_pts[attribute]:
            precision, less_than_category, greater_than_category = best_precision_with_weight(data, attribute, dvd_pt,
                                                                                              weight)
            precisions[attribute].append(precision)
            classify_results[attribute].append([less_than_category, greater_than_category])
    best_precision = -1
    # 找最高精度，划分点序数，和分类属性
    for attribute in precisions.keys():
        i = precisions[attribute].index(max(precisions[attribute]))
        if best_precision < precisions[attribute][i]:
            best_precision = precisions[attribute][i]
            best_index = i
            best_attribute = attribute
    best_dvd_pt = dvd_pts[best_attribute][best_index]
    less_than_category, greater_than_category = classify_results[best_attribute][best_index]
    return best_attribute, best_dvd_pt, less_than_category, greater_than_category


def decision_stump_predict(data: pd.DataFrame, decision_stump: tuple):
    """
    使用决策树桩对数据集分类。

    Args:
        data (pd.DataFrame): 数据集
        decision_stump (tuple): 决策树桩

    Returns:
        predict_label (list): 分类标签
    """
    dvd_attribute, dvd_pt, less_than_category, greater_than_category = decision_stump
    predict_label = []
    for index, row in data.iterrows():
        if row[dvd_attribute] < dvd_pt:
            predict_label.append(less_than_category)
        else:
            predict_label.append(greater_than_category)
    return predict_label


def calculate_precision(predict_label: list, true_label: list):
    """
    计算分类精度。

    Args:
        predict_label (list): 分类标签
        true_label (list): 真实标签

    Returns:
        分类精度
    """
    wrong = 0
    for i in range(len(predict_label)):
        if predict_label[i] != true_label[i]:
            wrong += 1
    return 1 - wrong / len(predict_label)


def decision_stump_precision(data: pd.DataFrame, decision_stump: tuple):
    """
    使用决策树桩对数据集分类，并计算分类精度。

    Args:
        data (pd.DataFrame): 数据集
        decision_stump (tuple): 决策树桩

    Returns:
        精度。
    """
    predict_label = decision_stump_predict(data, decision_stump)
    return calculate_precision(predict_label, data['类别'].tolist())


def update_weight(weight: np.ndarray, alpha: np.float64, predict_label: list, true_label: list):
    """
    更新样本权重分布。

    Args:
        weight (np.ndarray): 样本权重（分布）
        alpha (np.float64): 分类器权重
        predict_label (list): 分类标签
        true_label (list): 真实标签

    Returns:
        updated_weight (np.ndarray): 更新后的样本权重分布
    """
    exp_factor = np.where(np.array(predict_label) == np.array(true_label), np.exp(-alpha), np.exp(alpha))
    updated_weight = np.array(weight) * exp_factor  # 更新样本分布
    updated_weight = updated_weight / np.sum(updated_weight)  # 归一化
    return updated_weight


def adaboost_fit(n_estimator: int, data: pd.DataFrame):
    """
    训练AdaBoost分类器，基学习器为决策树桩。

    Args:
        n_estimator (int): 基学习器个数
        data (pd.DataFrame): 数据集

    Returns:
        decision_stumps (list): 决策树桩列表
        alphas (list): 分类器权重列表
        weight (np.ndarray): 样本权重列表
    """
    decision_stumps = []
    alphas = []
    weight = np.ones((1, len(data))) / len(data)
    for i in range(n_estimator):
        decision_stump = decision_stump_fit(data, weight)
        predict_label = decision_stump_predict(data, decision_stump)
        epsilon = 1 - calculate_precision(predict_label, data['类别'].tolist())
        if epsilon > 0.5:
            i -= 1
            weight = update_weight(weight, alpha, predict_label, data['类别'].tolist())
            continue
        else:
            print('epsilon:', epsilon)  # 调试代码
        if epsilon == 0:
            alpha = 2
        else:
            alpha = 0.5 * np.log((1 - epsilon) / epsilon)
        weight = update_weight(weight, alpha, predict_label, data['类别'].tolist())
        decision_stumps.append(decision_stump)
        alphas.append(alpha)
    return decision_stumps, alphas, weight


def adaboost_predict(data: pd.DataFrame, decision_stumps: list, alphas: list):
    """
    使用AdaBoost对数据集分类。

    Args:
        data (pd.DataFrame): 数据集
        decision_stumps (list): 决策树桩列表
        alphas (list): 分类器权重列表

    Returns:
        predict_label (list): 分类标签
    """
    base_classifier_predict_labels = np.array(
        [decision_stump_predict(data, decision_stump) for decision_stump in decision_stumps])
    predict_label = np.sign(np.sum(base_classifier_predict_labels.T * np.array(alphas), axis=1))
    return predict_label


def plot_decision_boundary(decision_stumps: list, alphas: list, X: np.ndarray, y: np.ndarray,
                           axes: list = [0.2, 0.8, 0, 0.5], alpha: float = 0.5, contour: bool = True):
    """
    绘制分类结果。

    Args:
        decision_stumps (list): 决策树桩列表
        alphas (list): 分类器权重列表
        X (np.ndarray): 数据集（无标签）
        y (np.ndarray): 真实标签
        axes (list): 坐标轴范围
        alpha (float): 不透明度
        contour (bool): 是否绘制轮廓线

    Returns:
        None
    """
    fig = plt.figure(figsize=[21.33, 11.25])
    ax1 = fig.add_subplot(1, 1, 1)
    x1s = np.linspace(axes[0], axes[1], 100)
    x2s = np.linspace(axes[2], axes[3], 100)
    x1, x2 = np.meshgrid(x1s, x2s)
    X_new = np.c_[x1.ravel(), x2.ravel()]
    y_pred = adaboost_predict(pd.DataFrame(X_new, columns=['属性1', '属性2']), decision_stumps, alphas)
    y_pred = np.array(y_pred).reshape(x1.shape)
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


def loop_plot_result(data, _range):
    """
    绘制不同基分类器个数下的分类结果，并保存图片。

    Args:
        data (pd.DataFrame): 数据集
        _range (tuple): 基分类器个数范围

    Returns:
        None
    """
    X = data.to_numpy()[:, 0:data.shape[1] - 1]
    y = data.to_numpy()[:, -1]
    for i in range(_range[0], _range[1]):
        decision_stumps, alphas, weight = adaboost_fit(i, data)
        plot_decision_boundary(decision_stumps, alphas, X, y)
        plt.savefig("n={}".format(i), dpi=300, bbox_inches='tight')
    plt.show()


if __name__ == '__main__':
    data_set = data_init()
    decision_stumps, alphas, weight = adaboost_fit(14, data_set)
    predict_label = adaboost_predict(data_set, decision_stumps, alphas)
    pre = calculate_precision(predict_label, data_set['类别'].tolist())
    print('基分类器-决策树桩：', decision_stumps)
    print('精度：', pre)
    print('alpha：', alphas)
    print('权重：', weight.flatten().tolist())
    print('真实分类：', data_set['类别'].tolist())
    print('预测结果：', predict_label.astype(int).tolist())
    # loop_plot_result(data_set, (1, 21))

    X = data_set.to_numpy()[:, 0:data_set.shape[1] - 1]
    y = data_set.to_numpy()[:, -1]
    plot_decision_boundary(decision_stumps, alphas, X, y)
    plt.show()
