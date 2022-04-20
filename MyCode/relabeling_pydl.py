import numpy as np


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
    def __init__(self, path, u, v, w, x):
        self.path = path
        self.acc = None
        self.disc = None
        self.ratio = None
        self.u = u
        self.v = v
        self.w = w
        self.x = x

    def accuracy(self, n_zero, n_one):
        n = self.u + self.w
        p = self.v + self.x
        #TODO if p >= n:
        if self.value == 1:
            self.acc = n - p
            self.disc = (self.u + self.v) / n_one - (self.w + self.x) / n_zero
        else:
            self.acc = p - n
            self.disc = -(self.u + self.v) / n_one + (self.w + self.x) / n_zero

        if self.acc == 0:
            self.ratio = self.disc / 0.0000000000000000000000000000000000001
        else:
            self.ratio = self.disc / self.acc

    def __str__(self):
        return f"Path: {self.path} \naccuracy: {self.acc} \ndiscrimination: {self.disc} \nratio: {self.ratio} \ncontigency: \n{[self.u, self.v]}\n{[self.w, self.x]}"

    def __repr__(self):
        return f"{self.path}"


def leafs_to_relabel(tree, y, sensitive, n_zero, n_one, leafs, length, path=tuple()):
    if 'feat' in tree:
        tmp = path + ((tree['feat'], 'left'),)
        leafs_to_relabel(tree['left'], y, sensitive, n_zero, n_one, leafs, length, tmp)
        tmp = path + ((tree['feat'], 'right'),)
        leafs_to_relabel(tree['right'], y, sensitive, n_zero, n_one, leafs, length, tmp)
    else:
        #tree = copy.deepcopy(tree)
        tree["u"] = 0
        tree["v"] = 0
        tree["w"] = 0
        tree["x"] = 0
        for id in tree["transactions"]:
            if sensitive[id] == 1 and y[id] == 0:
                tree["u"] += 1
            if sensitive[id] == 1 and y[id] == 1:
                tree["v"] += 1
            if sensitive[id] == 0 and y[id] == 0:
                tree["w"] += 1
            if sensitive[id] == 0 and y[id] == 1:
                tree["x"] += 1
                """"""
        n = (tree["u"] + tree["w"])
        p = (tree["v"] + tree["x"])
        if tree["value"] == 1:
            tree["acc"] = n - p
            tree["disc"] = (tree["u"] + tree["v"]) / n_one - (tree["w"] + tree["x"]) / n_zero
        else:
            tree["acc"] = p - n
            tree["disc"] = -(tree["u"] + tree["v"]) / n_one + (tree["w"] + tree["x"]) / n_zero
        leaf = Leaf(path, tree["u"] / length, tree["v"] / length, tree["w"] / length, tree["x"] / length)
        # leaf = Leaf(path, tree["u"], tree["v"], tree["w"], tree["x"])
        leaf.value = tree["value"]
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


def relab(tree, y, y_pred, sensitive, e):
    disc_t = discrimination(y, y_pred, sensitive)
    cnt = np.unique(sensitive, return_counts=True)[1]
    # ℐ := { 𝑙 ∈ ℒ ∣ Δdisc 𝑙 < 0 }
    I = list()
    leafs_to_relabel(tree, y, sensitive, cnt[0], cnt[1], I, len(y))
    # 𝐿 := {}
    L = list()
    # while rem disc(𝐿) > 𝜖 do
    while rem_disc(disc_t, L, e) > e and I:
        # best l := arg max 𝑙∈ℐ∖𝐿 (disc 𝑙 /acc 𝑙 )
        best_l = I[0]
        for leaf in I:
            if leaf.ratio > best_l.ratio:
                best_l = leaf
        # 𝐿 := 𝐿 ∪ {𝑙}
        L.append(best_l)
        I.remove(best_l)
        #print(len(L))
        #print(rem_disc(disc_t, L, e))
    return L


def relab_leaf_limit(tree, y, y_pred, sensitive, leaf_limit):
    disc_t = discrimination(y, y_pred, sensitive)
    cnt = np.unique(sensitive, return_counts=True)[1]
    # ℐ := { 𝑙 ∈ ℒ ∣ Δdisc 𝑙 < 0 }
    I = list()
    leafs_to_relabel(tree, y, sensitive, cnt[0], cnt[1], I, len(y))
    # 𝐿 := {}
    L = list()
    # while rem disc(𝐿) > 𝜖 do
    while leaf_limit > 0 and I:
        # best l := arg max 𝑙∈ℐ∖𝐿 (disc 𝑙 /acc 𝑙 )
        best_l = I[0]
        for leaf in I:
            if leaf.ratio > best_l.ratio:
                best_l = leaf
        # 𝐿 := 𝐿 ∪ {𝑙}
        L.append(best_l)
        I.remove(best_l)
        leaf_limit -= 1
        # print(rem_disc(disc_t, L, e))
    return L


def browse_and_relab(tree, path, leaf):
    if path:
        p = path.pop(0)[1]
        browse_and_relab(tree[p], path, leaf)
    else:
        n = leaf.u + leaf.w
        p = leaf.v + leaf.x
        if p > n:
            tree['value'] = 0
        else:
            tree['value'] = 1
        """
        if tree['value'] == 1:
            tree['value'] = 0
        elif tree['value'] == 0:
            tree['value'] = 1
        """
