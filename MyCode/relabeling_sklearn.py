import copy

import numpy as np
import pandas as pd


def discrimination2(dataset):
    length = len(dataset)
    w2 = len(dataset[(dataset['Sensitive'] == 0) & (dataset['Class'] == 0) & (dataset['Pred'] == 1)]) / length
    x2 = len(dataset[(dataset['Sensitive'] == 0) & (dataset['Class'] == 1) & (dataset['Pred'] == 1)]) / length
    u2 = len(dataset[(dataset['Sensitive'] == 1) & (dataset['Class'] == 0) & (dataset['Pred'] == 1)]) / length
    v2 = len(dataset[(dataset['Sensitive'] == 1) & (dataset['Class'] == 1) & (dataset['Pred'] == 1)]) / length

    b = (len(dataset[(dataset['Sensitive'] == 1) & (dataset['Class'] == 0)]) + len(
        dataset[(dataset['Sensitive'] == 1) & (dataset['Class'] == 1)]))
    b = b / length
    b_not = (len(dataset[(dataset['Sensitive'] == 0) & (dataset['Class'] == 0)]) + len(
        dataset[(dataset['Sensitive'] == 0) & (dataset['Class'] == 1)]))
    b_not = b_not / length

    # w1 = len(dataset[(dataset['Sensitive'] == 0) & (dataset['Class'] == 0) & (dataset['Pred'] == 0)])/lenght
    # x1 = len(dataset[(dataset['Sensitive'] == 0) & (dataset['Class'] == 1) & (dataset['Pred'] == 0)])/lenght
    # u1 = len(dataset[(dataset['Sensitive'] == 1) & (dataset['Class'] == 0) & (dataset['Pred'] == 0)])/lenght
    # v1 = len(dataset[(dataset['Sensitive'] == 1) & (dataset['Class'] == 1) & (dataset['Pred'] == 0)])/lenght
    # print(w1+w2+x1+x2+u1+u2+v1+v2)
    print((w2, x2, u2, v2, b, b_not))

    return ((w2 + x2) / b_not) - ((u2 + v2) / b)


# 0.0033293050858906204
def discrimination(y, y_pred, sensitive):
    w2, x2, u2, v2, b, b_not = 0, 0, 0, 0, 0, 0
    for index in range(0, len(y)):
        if y_pred[index] == 1:
            if sensitive[index] == 0:
                if y[index] == 0:
                    w2 += 1
                elif y[index] == 1:
                    x2 += 1
            elif sensitive[index] == 1:
                if y[index] == 0:
                    u2 += 1
                elif y[index] == 1:
                    v2 += 1
        if sensitive[index] == 1:
            b += 1
        elif sensitive[index] == 0:
            b_not += 1

    # print((u2 + v2, w2 + x2, b, b_not))

    length = len(y)
    w2 = w2 / length
    x2 = x2 / length
    u2 = u2 / length
    v2 = v2 / length

    b = b / length
    b_not = b_not / length
    # print((u2, v2, w2, x2, b, b_not))

    return ((w2 + x2) / b_not) - ((u2 + v2) / b)


class Leaf:
    def __init__(self, path, node_id, u, v, w, x, transactions=None):
        self.path = path
        self.node_id = node_id
        self.acc = None
        self.disc = None
        self.ratio = None
        self.u = u
        self.v = v
        self.w = w
        self.x = x
        self.transactions = transactions

    def accuracy(self, n_zero, n_one):
        n = self.u + self.w
        p = self.v + self.x
        if p >= n:
            self.acc = n - p
            self.disc = (self.u + self.v) / n_one - (self.w + self.x) / n_zero
        else:
            self.acc = p - n
            self.disc = -(self.u + self.v) / n_one + (self.w + self.x) / n_zero

        self.ratio = self.disc / self.acc

    def __str__(self):
        return f"Path: {self.path} \naccuracy: {self.acc} \nnode_id: {self.node_id} \ndiscrimination: {self.disc} \nratio: {self.ratio} " \
               f"\ncontigency: \n{[self.u, self.v]}\n{[self.w, self.x]}" \
               f"\ntransactions: {self.transactions}"

    def __repr__(self):
        return f"{self.path}"


def get_transactions(path, x):
    filtered = pd.DataFrame(x)
    for e in path:
        col = e[0]
        if e[1] == 'left':
            cond = 0
        else:
            cond = 1
        filtered = filtered.loc[filtered[col] == cond]
    return list(filtered.index)


"""
def leafs_to_relabel(tree, x, y, sensitive, n_zero, n_one, leafs, length, current, path=tuple(), level=0):
    id = tree.children_left[level]
    feature = tree.feature[id]
    tmp = path + ((current, 'left', level, id),)
    if feature == -2:
        transactions = get_transactions(tmp, x)
        leaf = Leaf(tmp, id, 0, 0, 0, 0, transactions)
        leaf.value = tree.value[id]
        for id in transactions:
            if sensitive[id] == 1 and y[id] == 0:
                leaf.u += 1
            if sensitive[id] == 1 and y[id] == 1:
                leaf.v += 1
            if sensitive[id] == 0 and y[id] == 0:
                leaf.w += 1
            if sensitive[id] == 0 and y[id] == 1:
                leaf.x += 1
        leaf.u = leaf.u / length
        leaf.v = leaf.v / length
        leaf.w = leaf.w / length
        leaf.x = leaf.x / length
        leaf.accuracy(n_zero / length, n_one / length)
        leafs.append(leaf)
    else:
        leafs_to_relabel(tree, x, y, sensitive, n_zero, n_one, leafs, length, feature, tmp, level + 1)

    id = tree.children_right[level]
    feature = tree.feature[id]
    tmp = path + ((current, 'right', level, id),)
    if feature == -2:
        transactions = get_transactions(tmp, x)
        leaf = Leaf(tmp, id, 0, 0, 0, 0, transactions)
        leaf.value = tree.value[id]
        for id in transactions:
            if sensitive[id] == 1 and y[id] == 0:
                leaf.u += 1
            if sensitive[id] == 1 and y[id] == 1:
                leaf.v += 1
            if sensitive[id] == 0 and y[id] == 0:
                leaf.w += 1
            if sensitive[id] == 0 and y[id] == 1:
                leaf.x += 1
        leaf.u = leaf.u / length
        leaf.v = leaf.v / length
        leaf.w = leaf.w / length
        leaf.x = leaf.x / length
        leaf.accuracy(n_zero / length, n_one / length)
        leafs.append(leaf)
    else:
        leafs_to_relabel(tree, x, y, sensitive, n_zero, n_one, leafs, length, feature, tmp, level + 1)
"""


def leafs_to_relabel(tree, x, y, sensitive, n_zero, n_one, leafs, length, node_id, path=tuple()):
    feature = tree.feature[node_id]
    if feature >= 0:
        tmp = path + ((feature, 'left', node_id),)
        leafs_to_relabel(tree, x, y, sensitive, n_zero, n_one, leafs, length, tree.children_left[node_id], tmp)
        tmp = path + ((feature, 'right', node_id),)
        leafs_to_relabel(tree, x, y, sensitive, n_zero, n_one, leafs, length, tree.children_right[node_id], tmp)
    else:
        transactions = get_transactions(path, x)
        tmp = path + ((feature, 'leaf', node_id),)

        u, v, w, x =0, 0, 0, 0
        for id in transactions:
            if sensitive[id] == 1 and y[id] == 0:
                u += 1
            if sensitive[id] == 1 and y[id] == 1:
                v += 1
            if sensitive[id] == 0 and y[id] == 0:
                w += 1
            if sensitive[id] == 0 and y[id] == 1:
                x += 1
        leaf = Leaf(tmp, node_id, u/length, v/length, w/length, x/length, transactions)
        leaf.value = tree.value[node_id]
        leaf.accuracy(n_zero / length, n_one / length)
        if leaf.disc < 0:
            leafs.append(leaf)


# rem disc(𝐿) := disc 𝑇 + ∑ Δdisc 𝑙 ≤ 𝜖
def rem_disc(disc_t, L, e):
    s = 0
    for leaf in L:
        if leaf.disc < e:
            s += leaf.disc
    return disc_t + s


def relab(tree, x, y, y_pred, sensitive, e):
    disc_t = discrimination(y, y_pred, sensitive)
    cnt = np.unique(sensitive, return_counts=True)[1]
    # ℐ := { 𝑙 ∈ ℒ ∣ Δdisc 𝑙 < 0 }
    I = list()
    leafs_to_relabel(tree, x, y, sensitive, cnt[0], cnt[1], I, len(y), 0)
    # 𝐿 := {}
    L = set()
    # while rem disc(𝐿) > 𝜖 do
    while rem_disc(disc_t, L, e) > e and I:
        # best l := arg max 𝑙∈ℐ∖𝐿 (disc 𝑙 /acc 𝑙 )
        best_l = I[0]
        for leaf in I:
            if leaf.ratio > best_l.ratio:
                best_l = leaf
        # 𝐿 := 𝐿 ∪ {𝑙}
        L.add(best_l)
        I.remove(best_l)

    return L


def browse_and_relab(clf, node_id):
    clf.tree_.value[node_id][0][0], clf.tree_.value[node_id][0][1] = clf.tree_.value[node_id][0][1], \
                                                                     clf.tree_.value[node_id][0][0]
