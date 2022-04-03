import copy
import numpy as np
import pandas as pd

def discrimination_dataset(y, sensitive):
    """
    p0: ∣{𝑥 ∈ 𝐷 ∣ 𝑥.Sensitive = 0, clf(𝑥) = +}∣
    p1: ∣{𝑥 ∈ 𝐷 ∣ 𝑥.Sensitive = 1, clf(𝑥) = +}∣
    n_zero: ∣{𝑥 ∈ 𝐷 ∣ 𝑥.Sensitive = 0}∣
    n_one: ∣{𝑥 ∈ 𝐷 ∣ 𝑥.Sensitive = 1}∣
    """
    p0, p1, n_zero, n_one = 0, 0, 0, 0
    for i in range(0, len(y)):
        if sensitive[i] == 0.0:
            n_zero += 1
            if y[i] == 1.0:
                p0 += 1
        elif sensitive[i] == 1.0:
            n_one += 1
            if y[i] == 1.0:
                p1 += 1

    if n_one == 0 and n_zero == 0:
        d = 0
    elif n_zero == 0:
        d = -(p1 / n_one)
    elif n_one == 0:
        d = p0 / n_zero
    else:
        d = (p0 / n_zero) - (p1 / n_one)
    return d

def discrimination(y, y_pred, sensitive):
    """

    :param y: The target sample
    :param y_pred: The target sample predicted by the decision tree
    :param sensitive: The sensitive sample
    :return: The discrimination of dataset
    """
    w2, x2, u2, v2, b, b_not = 0, 0, 0, 0, 0, 0
    y_length = len(y)
    for index in range(0, y_length):
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

    w2 = w2 / y_length
    x2 = x2 / y_length
    u2 = u2 / y_length
    v2 = v2 / y_length

    b = b / y_length
    b_not = b_not / y_length

    return ((w2 + x2) / b_not) - ((u2 + v2) / b)


class Leaf:
    """

    :param path: the threshold operator, can be either '>' or '<'
    :type path: str
    :param node_id:
    :type node_id: int
    :param u: The portion of item of the dataset whose class is negative and the sensitive attribute is positive
            contained by leaf
    :type u: float
    :param v: The portion of item of the dataset whose class is positive and the sensitive attribute is positive
            contained by leaf
    :type v: float
    :param w: The portion of item of the dataset whose class is negative and the sensitive attribute is negative
            contained by leaf
    :type w: float
    :param x: The portion of item of the dataset whose class is positive and the sensitive attribute is negative
            contained by leaf
    :type x: float
    :param transactions: A list of sample indexes used by the leaf
    :type transactions: list
    """

    def __init__(self, path, node_id, u, v, w, x, transactions=None, t=None):#TODO
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
        self.t = t
        #TODO
    def accuracy(self, cnt_p, cnt_n, portion_zero, portion_one):
        n = self.u + self.w
        p = self.v + self.x
        """"
        WARNING ! Don't use '(self.u + self.w) > (self.v + self.x)' or 'p>n'
        self.u, self.w,... are fractions, so in some cases this is not precise and causes a bug. 
        (can be caused by python rounding during a division)
        cnt_p and cnt_n are the number of positive and negative class, 
        thus integers, there will be no error when using them.
        """
        if cnt_p > cnt_n:
            self.acc = n - p
            self.disc = (self.u + self.v) / portion_one - (self.w + self.x) / portion_zero
            self.d = f"({self.t[0] + self.t[1]}) / {self.t[5]} - ({self.t[2] + self.t[3]}) / {self.t[4]}" \
                     f"= {(self.t[0] + self.t[1]) / self.t[5] - (self.t[2] + self.t[3]) / self.t[4]}"
            #TODO
        else:
            self.acc = p - n
            self.disc = -(self.u + self.v) / portion_one + (self.w + self.x) / portion_zero
            self.d = f"-({self.t[0]} + {self.t[1]}) / {self.t[5]} + ({self.t[2]} + {self.t[3]}) / {self.t[4]}" \
                     f"= {-(self.t[0] + self.t[1]) / self.t[5] + (self.t[2] + self.t[3]) / self.t[4]}"
            # TODO
        if self.acc == 0:
            """
            In theory, if the accuracy is 0 and the discrimination after relabeling (self.disc) is < 0, 
            this leaf must be one of the best to relabeling because 
            we will have a loss in discrimination but no loss in accuracy.
            This is why a positive value very close to 0 is used to avoid a division by 0 and to maintain a high ratio.
            """
            self.ratio = self.disc / -0.00000000000000000000000000000000000001
        else:
            self.ratio = self.disc / self.acc

    def __str__(self):
        return f"Path: format -> (feature, type, node id)\n{self.path} " \
               f"\nnode_id: {self.node_id} " \
               f"\nThe effect of relabeling the leaf on accuracy: {self.acc}" \
               f"\nThe effect of relabeling the leaf on discrimination: {self.disc} " \
               f"\n{self.d} " \
               f"\nratio: {self.ratio} " \
               f"\ncontingency table: \n{[self.u, self.v]}\n{[self.w, self.x]}" \
               f"\ntransactions: {self.transactions}"

    def __repr__(self):
        return f"{self.path}"


def get_transactions_by_leaf(clf, path, x):
    """

    :param path: A list of tuples representing a path to a leaf from the root node where a tuple is a leaf in the tree.
            The tuple is in the format (feature, type, node id). Feature = the feature of a leaf.
            Type allows to know if we have to go left or right when we navigate in the tree.
            Node id: id of the node in sklearn
    :param x: The sample used to train the model
    :return: A list of sample indexes used by the leaf
    """
    filtered = pd.DataFrame(x)
    for tupl in path:
        feature = tupl[0]
        node_id = tupl[2]
        if tupl[1] == 'left':
            filtered = filtered.loc[filtered[feature] <= clf.tree_.threshold[node_id]]
        elif tupl[1] == 'right':
            filtered = filtered.loc[filtered[feature] > clf.tree_.threshold[node_id]]
        else:
            raise Exception("Should not reach here")
    return list(filtered.index)


def get_leaves_candidates(clf, x, y, sensitive, cnt, length, leaves, node_id=0, path=tuple()):
    """

    :param clf: The decision tree classifier
    :param x: The input sample
    :param y: The target sample
    :param sensitive: The sensitive sample
    :param cnt: Tuple where the index 0 is the number of negative sensitive classes and
                at index 1 is the number of positive sensitive classes
    :param length: The size of the sample
    :param leaves: The leaves that could be used for the relabeling
    :param node_id: The identifier of the node we are in when we traverse the tree.
    :param path: A list of tuples representing a path to a leaf from the root node
    """
    feature = clf.tree_.feature[node_id]
    if feature >= 0:
        tmp_path = path + ((feature, 'left', node_id),)
        get_leaves_candidates(clf, x, y, sensitive, cnt, length, leaves, clf.tree_.children_left[node_id], tmp_path)
        tmp_path = path + ((feature, 'right', node_id),)
        get_leaves_candidates(clf, x, y, sensitive, cnt, length, leaves, clf.tree_.children_right[node_id], tmp_path)
    else:
        transactions = get_transactions_by_leaf(clf, path, x)
        tmp_path = path + ((feature, 'leaf', node_id),)

        u, v, w, x = 0, 0, 0, 0
        for transaction in transactions:
            if sensitive[transaction] == 1:
                if y[transaction] == 0:
                    u += 1
                elif y[transaction] == 1:
                    v += 1
            elif sensitive[transaction] == 0:
                if y[transaction] == 0:
                    w += 1
                elif y[transaction] == 1:
                    x += 1
        leaf = Leaf(tmp_path, node_id, u / length, v / length, w / length, x / length, transactions, (u,v,w,x, cnt[0], cnt[1]))
        leaf.value = copy.deepcopy(clf.tree_.value[node_id])
        leaf.accuracy(v + x, u + w, cnt[0] / length, cnt[1] / length)
        if leaf.disc < 0:
            leaves.append(leaf)


# rem disc(𝐿) := disc 𝑇 + ∑ Δdisc 𝑙 ≤ 𝜖
def rem_disc(disc_tree, leaves, threshold):
    """

    :param disc_tree: The discrimination of the tree
    :param leaves: The leaves that we will keep to relabel them
    :param threshold: The threshold of discrimination that we do not want to exceed
    :return: The new discrimination we will get
    """
    s = 0
    for leaf in leaves:
        if leaf.disc < threshold:
            s += leaf.disc
    return disc_tree + s


def leaves_to_relabel(clf, x, y, y_pred, sensitive, threshold):
    """

    :param clf: The decision tree
    :param x: The input sample
    :param y: The target sample
    :param y_pred: The target sample predicted by the decision tree
    :param sensitive: The sensitive sample
    :param threshold: The threshold of discrimination that we do not want to exceed
    :return: The leaves that we will keep to relabel them
    """
    disc_tree = discrimination(y, y_pred, sensitive)
    cnt = np.unique(sensitive, return_counts=True)[1]

    # ℐ := { 𝑙 ∈ ℒ ∣ Δdisc 𝑙 < 0 }
    i = list()
    get_leaves_candidates(clf, x, y, sensitive, cnt, len(y), i)
    # 𝐿 := {}
    leaves = set()
    # while rem disc(𝐿) > 𝜖 do
    while rem_disc(disc_tree, leaves, threshold) > threshold and i:
        # best l := arg max 𝑙∈ℐ∖𝐿 (disc 𝑙 /acc 𝑙 )
        best_l = i[0]
        for leaf in i:
            if leaf.ratio > best_l.ratio:
                best_l = leaf
        # 𝐿 := 𝐿 ∪ {𝑙}
        leaves.add(best_l)
        i.remove(best_l)

    if rem_disc(disc_tree, leaves, threshold) > threshold:
        print("\033[1;33m"+"Unable to reach the threshold."+"\033[0m")

    return leaves


def relab_leaf_limit(clf, x, y, y_pred, sensitive, leaf_limit):
    cnt = np.unique(sensitive)
    # ℐ := { 𝑙 ∈ ℒ ∣ Δdisc 𝑙 < 0 }
    i = list()
    get_leaves_candidates(clf, x, y, sensitive, cnt, len(y), i)
    # 𝐿 := {}
    l = set()
    # while rem disc(𝐿) > 𝜖 do
    while leaf_limit > 0 and i:
        # best l := arg max 𝑙∈ℐ∖𝐿 (disc 𝑙 /acc 𝑙 )
        best_l = i[0]
        for leaf in i:
            if leaf.ratio > best_l.ratio:
                best_l = leaf
        # 𝐿 := 𝐿 ∪ {𝑙}
        l.add(best_l)
        i.remove(best_l)

    return l


def browse_and_relab(clf, node_id):
    if clf.tree_.value[node_id][0][0] == clf.tree_.value[node_id][0][1]:
        clf.tree_.value[node_id][0][1] += 1
        return
    clf.tree_.value[node_id][0][0], clf.tree_.value[node_id][0][1] = clf.tree_.value[node_id][0][1], \
                                                                     clf.tree_.value[node_id][0][0]


def relabeling(clf, x, y, y_pred, sensitive, threshold):
    if discrimination_dataset(y, sensitive) < 0:
        print(discrimination_dataset(y, sensitive))
        raise Exception("The discrimination of the dataset can't be negative.")
    if len(np.unique(sensitive)) != 2:
        raise Exception("Only two different labels are expected for the sensitive sample.")
    if len(np.unique(y)) != 2:
        raise Exception("Only two different labels are expected for the target sample.")

    for leaf in leaves_to_relabel(clf, x, y, y_pred, sensitive, threshold):
        if clf.tree_.value[leaf.node_id][0][0] == clf.tree_.value[leaf.node_id][0][1]:
            clf.tree_.value[leaf.node_id][0][1] += 1
            return
        clf.tree_.value[leaf.node_id][0][0], clf.tree_.value[leaf.node_id][0][1] = clf.tree_.value[leaf.node_id][0][1], \
                                                                                   clf.tree_.value[leaf.node_id][0][0]


def relabeling_leaf_limit(clf, x, y, y_pred, sensitive, leaf_limit):
    for leaf in relab_leaf_limit(clf, x, y, y_pred, sensitive, leaf_limit):
        if clf.tree_.value[leaf.node_id][0][0] == clf.tree_.value[leaf.node_id][0][1]:
            clf.tree_.value[leaf.node_id][0][1] += 1
            return
        clf.tree_.value[leaf.node_id][0][0], clf.tree_.value[leaf.node_id][0][1] = clf.tree_.value[leaf.node_id][0][1], \
                                                                                   clf.tree_.value[leaf.node_id][0][0]
