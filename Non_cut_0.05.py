from pathlib import Path
from typing import *

import networkx as nx
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from tqdm.auto import tqdm

import rpy2.robjects as robjects
import rpy2.robjects.numpy2ri
from rpy2.robjects.packages import importr

l0learn = importr("L0Learn")
ncvreg = importr("ncvreg")
rpy2.robjects.numpy2ri.activate()

def graph_from_pandas(file_path):
    df = pd.read_excel(file_path, usecols=[1, 2, 3])
    df.columns = ["source", "target", "value"]

    le = LabelEncoder()
    le.fit(pd.concat([df[col] for col in df.columns[:2]]))
    for col in df.columns[:2]:
        df[col] = le.transform(df[col])

    g = nx.from_pandas_edgelist(df, edge_attr="value")
    diff = set(df["source"]) - set(df["target"])
    if len(diff) == 1:
        source = diff.pop()
    return g, source

from dataclasses import dataclass


@dataclass
class Tree:
    tree: nx.DiGraph
    source: int

    def __init__(self, g, source):
        assert nx.is_tree(g)
        tree = nx.dfs_tree(g, source=source).to_directed()
        for edge in tqdm(tree.edges()):
            tree.edges[edge].update({"value": g.get_edge_data(*edge)["value"]})
        self.tree = tree
        self.source = source
        # self.reduce()
        self.leaves = [n for n in self.tree.nodes() if self.tree.out_degree(n) == 0]
        assert self.source not in self.leaves
        self.parent_dict = {self.source: None}
        for node in self.tree.nodes():
            if node != self.source:
                parent = next(self.tree.predecessors(node))
                self.parent_dict[node] = parent

        self.edges = list(nx.bfs_edges(self.tree, source=self.source))
        print(self.edges, "kk")
        print(len(self.edges), "jjj")
        # self.edges = list(set(nx.dfs_edges(self.tree, source=self.source)))
        self.edge2idx = {
            edge: idx for edge, idx in zip(self.edges, range(len(self.edges)))
        }
        self.idx2edge = {v: k for k, v in self.edge2idx.items()}
        cut_paths, cut_map, cut_edge2idx = self.find_cut_paths()
        self.cut_edge2idx = cut_edge2idx
        print(cut_edge2idx)

    def find_cut_paths(self):
        leaves = [n for n in self.tree.nodes() if self.tree.out_degree(n) == 0]
        parent_dict = {self.source: None}
        for node in self.tree.nodes():
            if node != self.source:
                parent = next(self.tree.predecessors(node))
                parent_dict[node] = parent
        cut_paths = []
        cut_map = {}  # 新增一个字典
        cut_edge2idx = {}
        idx = 0  # 新增一个变量，用来记录index
        for leaf in tqdm(leaves):
            cut_path = []
            node = leaf
            last_visit = None
            while node is not None:
                if self.tree.in_degree(node) == 1 and self.tree.out_degree(node) == 1:
                    if not cut_path:
                        cut_path.append(last_visit)
                    cut_path.append(node)
                else:
                    if cut_path:
                        cut_path.append(next(self.tree.predecessors(cut_path[-1])))
                        cut_path.reverse()
                        cut_paths.append(tuple(cut_path))
                        # 新增一行代码，将切割路径和新边加入到字典中
                        cut_map[tuple((i, o) for i, o in zip(cut_path[:-1], cut_path[1:]))] = (
                            cut_path[0], cut_path[-1])
                        # 修改一段代码，将cut_map中的值作为键，加入到cut_edge2idx中
                        new_edge = cut_map[tuple((i, o) for i, o in zip(cut_path[:-1], cut_path[1:]))]
                        cut_edge2idx[new_edge] = idx
                        idx += 1
                        cut_path = []
                    else:
                        if last_visit is not None and node is not None:
                            edge = (last_visit, node)
                            cut_map[(edge,)] = edge  # 将边作为一个单元素的元组作为键，值为边本身
                            new_edge = cut_map[(edge,)]
                            cut_edge2idx[new_edge] = idx
                            idx += 1
                last_visit = node
                node = parent_dict[node]

        return list(set(cut_paths)), cut_map, cut_edge2idx

    def get_Ax_raw(self):
        mat = []
        for leaf in tqdm(self.leaves):
            path = []
            node = leaf
            while node is not None:
                path.append(node)
                node = self.parent_dict[node]
            path.reverse()
            path_edges = [(i, o) for i, o in zip(path[:-1], path[1:])]
            repr = np.zeros(len(self.edges))
            for edge in path_edges:
                repr[self.edge2idx[edge]] = 1
            mat.append(repr)

        A = np.asarray(mat)
        x = [nx.get_edge_attributes(self.tree, "value").values()]
        x = np.asarray(list(x[0]))
        return A, x

    def get_subnodes(self, edges):
        nodes = []
        for edge in edges:
            nodes.extend(self.edges[edge])
        expanded_nodes = list(set(nodes))
        print(expanded_nodes)
        for node in nodes:
            expanded_nodes.extend(list(self.tree.predecessors(node)))
        return list(set(expanded_nodes))

def is_equal(a, b):
    return jnp.all(a == b)


def quotient(array, is_equal):
    equal_array_mat = vmap(is_equal, in_axes=(None, 0), out_axes=0)
    equal_mat_mat = vmap(equal_array_mat, in_axes=(0, None), out_axes=0)

    arrays = jnp.array_split(array, 256)

    ress = [equal_mat_mat(a, array) for a in tqdm(arrays)]
    res = jnp.concatenate(ress)
    res.shape

    whrs = [np.where(a)[0] for a in tqdm(res)]

    drop_list = []
    for i, wh in tqdm(enumerate(whrs)):
        if i not in drop_list and len(wh) > 1:
            drop_list.extend([t for t in wh if t != i])
    # drop_list

    e_classes = [whrs[i] for i in tqdm(range(len(whrs))) if i not in drop_list]
    return e_classes


def get_sum(equivalent_class):
    return np.sum([x[i] for i in equivalent_class])



def linear_log(x):
    return -np.log(x) if x > 0 else 0


class Evaluator:
    def __init__(self, x, cut_map, tree):
        self.x = x
        self.cut_map = cut_map
        self.tree = tree

    def evaluateAcc(self, x_hat):
        res = pd.DataFrame([x_hat, x]).T
        pd.set_option('display.max_rows', None)
        pre_Acc = []
        for cut_path, new_edge in self.cut_map.items():
            # 如果是无法融合的边，直接判断预测值和真实值的差值是否小于500
            if len(cut_path) == 1 and cut_path[0] == new_edge:
                edge = cut_path[0]
                print(type(edge))
                edge = tuple(reversed(edge))
                # 使用self.tree.edge2idx[edge]来获取边对应的索引
                temp = self.tree.edge2idx[edge]
                if abs(res.loc[temp][1] - res.loc[temp][0]) <= 500:
                    pre_Acc.append(edge)
            # 如果是切割路径上的边，判断每一条边的预测值和真实值的差值是否小于500
            else:
                sum_true = 0
                sum_pred = 0
                for edge in cut_path:
                    # 使用self.tree.edge2idx[edge]来获取边对应的索引
                    sum_true += res.loc[self.tree.edge2idx[edge]][1]
                    sum_pred += res.loc[self.tree.edge2idx[edge]][0]
                # 使用self.tree.edge2idx[new_edge]来获取新边对应的索引
                new_temp = self.tree.cut_edge2idx[new_edge]

                print(self.tree.cut_edge2idx, "new")
                # 修改一段代码，将if语句中的条件改成一个for循环，遍历切割路径上的每一条边，然后判断每一条边的预测值和真实值的差值是否小于500，如果是，就将这条边加入到pre_Acc列表中
                for edge in cut_path:  # 遍历切割路径上的每一条边
                    temp = self.tree.edge2idx[edge]  # 获取这条边在原始树中对应的索引
                    if abs(res.loc[temp][1] - res.loc[temp][0]) <= 500:  # 如果这条边的真实值和预测值的差值的绝对值小于等于500
                        pre_Acc.append(edge)  # 将这条边加入到pre_Acc列表中

        acc = len(pre_Acc) / len(res[0])

        return {"Accuracy": acc}

    def evaluateCov(self, x_hat, index):
        res = pd.DataFrame([x_hat, x]).T
        pd.set_option('display.max_rows', None)

        #         real_nonzero = res[abs(res[1]) > 20]
        real_nonzero = res.loc[index]
        pre_Acc = real_nonzero[abs(real_nonzero[1] - real_nonzero[0]) <= 500]
        # print(pre_Acc)
        cov = len(pre_Acc) / len(real_nonzero)

        return {"CoverAcc": cov}

file_path = Path("./时间_跟踪拓扑单棵树TXT展示_2_树的连接边.xlsx")

g, source = graph_from_pandas(file_path)

tree = Tree(g, source)

A, x = tree.get_Ax_raw()
print(A.shape)
# print(A.shape)
metric1_1 = list()
metric2_1 = list()

metric1_2 = list()
metric2_2 = list()
cut_paths, cut_map, cut_edge2idx = tree.find_cut_paths()

ratio = 0.3
for j in range(10):
    print(j)
    x = np.zeros(A.shape[1])
    leng = len(x)

    np.random.seed(j)

    neg_index = np.random.choice(np.arange(leng), size=int(leng * ratio))
    random_numbers1 = np.random.normal(loc=0, scale=2000000 / 3, size=len(neg_index))
    print(len(random_numbers1))
    for i in range(len(neg_index)):
        ind = neg_index[i]
        x[ind] = x[ind] + random_numbers1[i]

    random_numbers2 = np.random.normal(loc=0, scale= 20 / 3, size=leng)
    for i in range(leng):
        x[i] = x[i] + random_numbers2[i]
    #     print(sum(x))
    x_ori = x

    b = A @ x

    random_numbers3 = np.random.normal(loc=250, scale=350 / 3, size=leng)
    for i in range(len(b)):
        b[i] = b[i] + random_numbers3[i]
#
#
#     #     model = SCAD.msasnet(A,b,family = "gaussian" ,init = "snet" ,tune='bic',tune_nsteps = 'ebic',alphas = np.arange(0.05, 0.9, 0.05))
#     #     coef_fun = robjects.r["coef"]
#     #     x_hat = coef_fun(model)
#
    b = list(b)
#
    #     model = ncvreg.cv_ncvreg(A,b,family = "gaussian" ,penalty = "SCAD",lambda_min=0.0001)
    #     minlambda = dict(model.items())['min']
    #     model = ncvreg.ncvreg(A,b,family = "gaussian" ,penalty = "SCAD",lambda_min=0.00001)
    #     beta = dict(model.items())["beta"]
    # #     mincol = minlambda-1
    #     x_hat = beta[1:,99]

    model = ncvreg.ncvreg(A, b, family="gaussian", penalty="SCAD", lambda_min=0.00001)
    beta = dict(model.items())["beta"]
    #     mincol = minlambda-1
    x_hat = beta[1:, 99]

    """
    Evaluate
    """
    x_hat = np.squeeze(x_hat)
    metrics1 = Evaluator(x_ori, cut_map, tree).evaluateAcc(x_hat)
    metrics2 = Evaluator(x_ori, cut_map, tree).evaluateCov(x_hat, neg_index)

    #     print(metrics1["Accuracy"])
    #     print(metrics2['CoverAcc'],"\n")

    metric1_2.append(metrics1["Accuracy"])
    metric2_2.append(metrics2['CoverAcc'])

print("Average")
print("####################")
# print("L0learn")
# print(np.mean(metric1_1))
# print(np.mean(metric2_1))
print("SCAD")
print(np.mean(metric1_2))
print(np.mean(metric2_2))

print("\nMin")
print("####################")
# print("L0learn")
# print(min(metric1_1))
# print(min(metric2_1))
print("SCAD")
print(min(metric1_2))
print(min(metric2_2))
#
# test = pd.DataFrame({'acc': metric1_1, 'cov': metric2_1})
# test.to_csv('0.1ra-l0learn.csv', encoding='gbk')
#
# test = pd.DataFrame({'acc': metric1_2, 'cov': metric2_2})
# test.to_csv('0.05ra-SCAD_cut.csv', encoding='gbk')