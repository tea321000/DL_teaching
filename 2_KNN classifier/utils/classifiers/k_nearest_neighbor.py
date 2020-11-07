from builtins import range
from builtins import object
import numpy as np
from past.builtins import xrange


class KNearestNeighbor(object):
    """ L2范数的KNN分类器 """

    def __init__(self):
        pass

    def train(self, X, y):
        """
        训练分类器。对KNN来说仅仅是把训练集数据保存即可。

        输入:
        - X: 形状为 (num_train, D) 的numpy数组，训练集样本数量为 num_train ，维数为 D。
        - y: 形状为 (N,) 的numpy数组，y[i]是训练集样本x[i]的标签。
        """
        self.X_train = X
        self.y_train = y

    def predict(self, X, k=1, num_loops=0):
        """
        使用分类器对测试集图片进行分类。

        输入:
        - X: 形状为 (num_test, D) 的numpy数组，测试集样本数量为 num_test ，样本维数为 D。
        - k: 进行多数投票决定测试集图片分类的邻居数量。
        - num_loops: 选择使用哪种方式计算训练集样本和测试集样本的距离。

        返回:
        - y: 形状为 (num_test,) 的numpy数组，y[i]是测试集样本x[i]的标签。
        """
        if num_loops == 0:
            dists = self.compute_distances_no_loops(X)
        elif num_loops == 1:
            dists = self.compute_distances_one_loop(X)
        elif num_loops == 2:
            dists = self.compute_distances_two_loops(X)
        else:
            raise ValueError('Invalid value %d for num_loops' % num_loops)

        return self.predict_labels(dists, k=k)

    def compute_distances_two_loops(self, X):
        """
        使用嵌套循环计算测试集样本X与训练集所有样本 self.X_train 的距离。

        输入:
        - X: 形状为 (num_test, D) 的numpy数组，测试集样本数量为 num_test ，样本维数为 D。

        返回:
        - dists: 形状为 (num_test, num_train) 的numpy数组， dists[i, j]是第i个测试样本和第j
        个训练样本的欧氏距离。
        """
        num_test = X.shape[0]
        num_train = self.X_train.shape[0]
        dists = np.zeros((num_test, num_train))
        for i in range(num_test):
            for j in range(num_train):
                #####################################################################
                # TODO:                                                             #
                # 计算第i个测试样本和第j个训练样本的l2距离，并保存在 dists[i, j] 中。你不能使用 #
                # 样本维数上的循环，也不能使用np.linalg.norm()内置函数。                    #
                #####################################################################
                # *****代码开始 (不要删除/修改该行)*****

                pass

                # *****代码结束 (不要删除/修改该行)*****
        return dists

    def compute_distances_one_loop(self, X):
        """
        使用单个循环计算测试集样本X与训练集所有样本 self.X_train 的距离。

        输入 / 输出: 与 compute_distances_two_loops 一致
        """
        num_test = X.shape[0]
        num_train = self.X_train.shape[0]
        dists = np.zeros((num_test, num_train))
        for i in range(num_test):
            #######################################################################
            # TODO:                                                               #
            # 计算第i个测试样本和第j个训练样本的l2距离，并保存在 dists[i, j] 中。           #
            # 不能使用 np.linalg.norm()内置函数。                                     #
            #######################################################################
            # *****代码开始 (不要删除/修改该行)*****

            pass

            # *****代码结束 (不要删除/修改该行)*****
        return dists

    def compute_distances_no_loops(self, X):
        """
        不使用显式循环计算测试集样本X与训练集所有样本 self.X_train 的距离。

        输入 / 输出: 与 compute_distances_two_loops 一致
        """
        num_test = X.shape[0]
        num_train = self.X_train.shape[0]
        dists = np.zeros((num_test, num_train))
        #########################################################################
        # TODO:                                                                 #
        # 不使用显式循环计算第i个测试样本和第j个训练样本的l2距离，并保存在 dists[i, j] 中。 #
        #                                                                       #
        # 你只能使用基本的数组操作来完成; 不能使用诸如 scipy 和 np.linalg.norm() 等内置的 #
        # 科学计算库以及内置函数。                                                   #
        #                                                                       #
        # 提示: 尝试使用矩阵乘法和广播机制（broadcast）来构建l2范数距离                  #
        #########################################################################
        # *****代码开始 (不要删除/修改该行)*****

        pass

        # *****代码结束 (不要删除/修改该行)*****
        return dists

    def predict_labels(self, dists, k=1):
        """
        给定测试样本和训练样本的距离矩阵，预测每个测试样本的类别标签。

        输入:
        - dists: 形状为 (num_test, num_train) 的numpy数组，dists[i, j] 为第i个测试样本与
        第j个训练样本之间的距离。

        返回:
        - y: 形状为 (num_test,) 的numpy数组，y[i]是测试集样本x[i]的标签。
        """
        num_test = dists.shape[0]
        y_pred = np.zeros(num_test)
        for i in range(num_test):
            # 用来保存第i个测试样本的k个最近邻居的长度为k的list。
            closest_y = []
            #########################################################################
            # TODO:                                                                 #
            # 使用距离矩阵来找到第i个测试样本的k个最近的邻居，并使用 self.y_train 来找到这些邻  #
            # 居的标签类别，保存这些标签类别到 closest_y 中。                              #
            # 提示: 查阅numpy.argsort的API                                            #
            #########################################################################
            # *****代码开始 (不要删除/修改该行)*****

            pass

            # *****代码结束 (不要删除/修改该行)*****
            #########################################################################
            # TODO:                                                                 #
            # 现在你已经可以找到k个最近邻居的标签，你需要通过 closest_y 出现频率最高的类别来决定 #
            # 测试样本的标签。保存标签在 y_pred[i] 中。                                  #
            #########################################################################
            # *****代码开始 (不要删除/修改该行)*****

            pass

            # *****代码结束 (不要删除/修改该行)*****

        return y_pred
