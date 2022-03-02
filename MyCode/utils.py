import ast
import math
import uuid

import numpy as np
from dl85.errors.errors import TreeNotFoundError, SearchFailedError
from matplotlib import pyplot as plt
from matplotlib.pyplot import suptitle
from sklearn.exceptions import NotFittedError


def train_test_split(random_state, X, y, sensitive):
    index_train = list(X.sample(frac=0.8, random_state=random_state).index)
    index_test = list(X.drop(index=index_train).index)

    X_train = X.drop(index=index_test).to_numpy()
    y_train = y.drop(index=index_test).to_numpy()
    sensitive_train = sensitive.drop(index=index_test).to_numpy()

    X_test = X.drop(index=index_train).to_numpy()
    y_test = list(y.drop(index=index_train).to_numpy())
    sensitive_test = sensitive.drop(index=index_train).to_numpy()

    return X_train, X_test, y_train, y_test, sensitive_train, sensitive_test


def discrimination(y, sensitive):
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

    # A higher discrimination means that tuples with
    # Sensitive = 1 are less likely to be classified as positive
    return d


def discr_add(tids, y, sensitive):
    """
    p0: ∣{𝑥 ∈ 𝐷 ∣ 𝑥.Sensitive = 0, 𝑥.Class = +}∣
    p1: ∣{𝑥 ∈ 𝐷 ∣ 𝑥.Sensitive = 1, 𝑥.Class = +}∣
    n_zero: ∣{𝑥 ∈ 𝐷 ∣ 𝑥.Sensitive = 0}∣
    n_one: ∣{𝑥 ∈ 𝐷 ∣ 𝑥.Sensitive = 1}∣
    """
    p0, p1 = 0, 0
    for i in tids:
        if sensitive[i] == 0.0:
            if y[i] == 1.0:
                p0 += 1
        elif sensitive[i] == 1.0:
            if y[i] == 1.0:
                p1 += 1

    cnt = np.unique(sensitive, return_counts=True)[1]
    n_zero = cnt[0]
    n_one = cnt[1]

    if n_one == 0 and n_zero == 0:
        d = 0
    elif n_zero == 0:
        d = -(p1 / n_one)
    elif n_one == 0:
        d = p0 / n_zero
    else:
        d = (p0 / n_zero) - (p1 / n_one)

    # A higher discrimination means that tuples with
    # Sensitive = 1 are less likely to be classified as positive
    return d


def misclassified(tids, y):
    classes, supports = np.unique(y.take(tids), return_counts=True)
    maxindex = np.argmax(supports)

    return sum(supports) - supports[maxindex], classes[maxindex]


def error(tids, k, y, sensitive):
    mis = misclassified(tids, y)
    return mis[0] + abs(discr_add(tids, y, sensitive)) * k, mis[1]


def tree_upgrade(tree, y, sensitive):
    if 'feat' in tree:
        tree_upgrade(tree['left'], y, sensitive)
        tree_upgrade(tree['right'], y, sensitive)
    else:
        tree['discrimination_additive'] = discr_add(tree['transactions'], y, sensitive)
        tree['misclassified'] = misclassified(tree['transactions'], y)[0]


plt.rcParams['figure.figsize'] = [9, 6]


def plot_mean(x_axe, y_axe, r, s1, s2):
    plt.figure(figsize=(9, 6))
    # style = ['solid', 'dotted', ':', '-.', 'dashed']
    style = ['solid', '–', '—', '-.', ':', '.', 'o', ',', 'v', '^']
    colors = ['#006400', '#00008b', '#b03060', '#ff4500', '#ffd700', '#7cfc00', '#00ffff', '#ff00ff', '#6495ed',
              '#ffdab9']
    fig, ax = plt.subplots()
    for k in r['k'].unique():
        y_values = list()
        for depth in r['depth'].unique():
            df1 = r.loc[(r["k"] == k) & (r["depth"] == depth)]
            y_values.append(df1[y_axe].mean())
        plt.plot(r['depth'].unique(), y_values, label="k=" + str(k), color=colors.pop(), linestyle=style[0])
    ax.set_xlabel(x_axe)
    ax.set_ylabel(y_axe)
    ax.legend()
    ax.set_ylim(s1, s2)
    plt.show()


def plot2(x_axe, y_axe, r):
    plt.figure(figsize=(9, 6))
    # r = r.loc[r["min_supp"] == 1]
    for k in r['k'].unique():
        x_values = list()
        y_values = list()

        for depth in r['depth'].unique():
            df1 = r.loc[(r["k"] == k) & (r["depth"] == depth)]
            y_values.append(df1[y_axe].mean())
            x_values.append(df1[x_axe].mean())
        plt.plot(x_values, y_values, label="k=" + str(k))

    plt.xlabel(x_axe)
    plt.ylabel(y_axe)
    plt.legend()

    plt.show()


def plot_one_scatter_by_depth(x_axe, y_axe, r, x_lim=None, y_lim=None):
    for k in r['k'].unique():
        # nrows = math.ceil(len(r['depth'].unique()) / 4)
        nrows = 2
        fig, axes = plt.subplots(nrows=nrows, ncols=4, figsize=(18, 9))
        ax_row = 0
        ax_col = 0
        for depth in r['depth'].unique():
            if ax_col == 4:
                ax_col = 0
                ax_row += 1
            if nrows == 1:
                ax = axes[ax_col]
            else:
                ax = axes[ax_row][ax_col]
            rr = r.loc[(r["k"] == k) & (r["depth"] == depth)]
            x = rr[x_axe].tolist()
            y = rr[y_axe].tolist()
            for i in range(0, len(x)):
                if y_lim is not None:
                    ax.set_ylim(y_lim[0], y_lim[1])
                if x_lim is not None:
                    ax.set_xlim(x_lim[0], x_lim[1])
                ax.scatter(x[i], y[i], s=10)
            ax.set_xlabel(x_axe)
            ax.set_ylabel(y_axe)
            ax.title.set_text('depth=' + str(depth))
            ax_col += 1
        # fig.delaxes(axes[1][3])
        plt.subplots_adjust(wspace=0.5, hspace=0.5)
        suptitle('k:' + str(k))
        plt.show()


def sum_elem_tree(tree, label, s=None, bool=True):
    if s is None:
        s = list()
    if 'feat' in tree:
        sum_elem_tree(tree['left'], label, s)
        sum_elem_tree(tree['right'], label, s)
    else:
        s.append(tree[label])
    if not bool:
        return sum(s)


from matplotlib.legend_handler import HandlerBase


class MarkerHandler(HandlerBase):
    def create_artists(self, legend, tup, xdescent, ydescent,
                       width, height, fontsize, trans):
        return [plt.Line2D([width / 2], [height / 2.], ls="",
                           marker=tup[1], color=tup[0], transform=trans)]


plt.rcParams['figure.figsize'] = [9, 6]


def plot_k_depth_mean(x_axe, y_axe, r, x_lim=None, y_lim=None):
    plt.figure(figsize=(9, 6))

    # markers = ['o', '^', '>', 'v', '<', '8', 's', 'p', '*', 'h', 'H', 'D', 'd', 'P', 'X']
    markers = ['o', 'x', 's', 'd', 'p', '8', 's', 'p', '*', 'h', 'H', 'D', 'd', 'P', 'X']
    fillstyles = ['full', 'left', 'right', 'bottom', 'top', 'none']
    colors = ['#006400', '#00008b', '#b03060', '#ff4500', '#ffd700', '#7cfc00', '#00ffff', '#ff00ff', '#6495ed',
              '#ffdab9']
    fig, ax = plt.subplots()
    index_color = 0
    for k in r['k'].unique():
        index_mark = 0
        for depth in r['depth'].unique():
            df1 = r.loc[(r["k"] == k) & (r["depth"] == depth)]
            ax.scatter(df1[x_axe].mean(), df1[y_axe].mean(), s=20, marker=markers[index_mark], c=colors[index_color])
            index_mark += 1
        index_color += 1

    ax.set_xlabel(x_axe)
    ax.set_ylabel(y_axe)
    list_color = colors + ["#000000"] * 5
    list_mak = ["o"] * 10 + markers

    list_lab = []
    for k in r['k'].unique():
        list_lab.append('k=' + str(k))
    for depth in r['depth'].unique():
        list_lab.append('Depth ' + str(depth))

    ax.legend(list(zip(list_color, list_mak)), list_lab,
              handler_map={tuple: MarkerHandler()}, ncol=3)
    if y_lim is not None:
        ax.set_ylim(y_lim[0], y_lim[1])
    if x_lim is not None:
        ax.set_xlim(x_lim[0], x_lim[1])
    plt.show()


def plot_each_k_depth_mean(x_axe, y_axe, r, x_lim=None, y_lim=None):
    # nrows = math.ceil(len(r['depth'].unique()) / 4)
    nrows = 2
    fig, axes = plt.subplots(nrows=nrows, ncols=4, figsize=(18, 9))

    colors = ['#006400', '#00008b', '#b03060', '#ff4500', '#ffd700', '#7cfc00', '#00ffff', '#ff00ff', '#6495ed',
              '#ffdab9']
    ax_row = 0
    ax_col = 0
    for depth in r['depth'].unique():
        if ax_col == 4:
            ax_col = 0
            ax_row += 1
        if nrows == 1:
            ax = axes[ax_col]
        else:
            ax = axes[ax_row][ax_col]
        index_color = 0
        for k in r['k'].unique():
            rr = r.loc[(r["k"] == k) & (r["depth"] == depth)]
            if y_lim is not None:
                ax.set_ylim(y_lim[0], y_lim[1])
            if x_lim is not None:
                ax.set_xlim(x_lim[0], x_lim[1])

            ax.scatter(rr[x_axe].mean(), rr[y_axe].mean(), s=10, c=colors[index_color])
            index_color += 1
        ax.set_ylabel(y_axe)
        ax.set_xlabel(x_axe)
        ax.title.set_text('depth= ' + str(depth))
        ax_col += 1
    plt.subplots_adjust(wspace=0.5, hspace=0.5)
    plt.show()


def get_dot_body(treedict, parent=None, left=True):
    gstring = ""
    id = str(uuid.uuid4())
    id = id.replace('-', '_')

    if "feat" in treedict.keys():
        feat = treedict["feat"]
        if parent is None:
            gstring += "node_" + id + " [label=\"{{feat|" + str(feat) + "}}\"];\n"
            gstring += get_dot_body(treedict["left"], id)
            gstring += get_dot_body(treedict["right"], id, False)
        else:
            gstring += "node_" + id + " [label=\"{{feat|" + str(feat) + "}}\"];\n"
            gstring += "node_" + parent + " -> node_" + id + " [label=" + str(int(left)) + "];\n"
            gstring += get_dot_body(treedict["left"], id)
            gstring += get_dot_body(treedict["right"], id, False)
    else:
        val = str(int(treedict["value"])) if treedict["value"] - int(treedict["value"]) == 0 else str(
            round(treedict["value"], 3))
        err = str(int(treedict["error"])) if treedict["error"] - int(treedict["error"]) == 0 else str(
            round(treedict["error"], 3))
        misclassified = str(int(treedict["misclassified"])) if treedict["misclassified"] - int(
            treedict["misclassified"]) == 0 else str(
            round(treedict["misclassified"], 3))
        discr = str(int(treedict["discrimination_additive"])) if treedict["discrimination_additive"] - int(
            treedict["discrimination_additive"]) == 0 else str(
            round(treedict["discrimination_additive"], 3))
        """
        true_pos = str(int(treedict["true_pos"])) if treedict["true_pos"] - int(
            treedict["true_pos"]) == 0 else str(
            round(treedict["true_pos"], 3))
        false_pos = str(int(treedict["false_pos"])) if treedict["false_pos"] - int(
            treedict["false_pos"]) == 0 else str(
            round(treedict["false_pos"], 3))
        true_neg = str(int(treedict["true_neg"])) if treedict["true_neg"] - int(
            treedict["true_neg"]) == 0 else str(
            round(treedict["true_neg"], 3))
        false_neg = str(int(treedict["false_neg"])) if treedict["false_neg"] - int(
            treedict["false_neg"]) == 0 else str(
            round(treedict["false_neg"], 3))
            """
        # maxi = max(len(val), len(err))
        # val = val if len(val) == maxi else val + (" " * (maxi - len(val)))
        # err = err if len(err) == maxi else err + (" " * (maxi - len(err)))
        gstring += "leaf_" + id + " [label=\"{{class|" + val + "}|{error|" + err + "}|{misclassified|" + misclassified + "}|{discrimination|" + discr \
                   + "}}\"];\n"
                   #+ "}|{true positive|" + true_pos + "}|{false positive|" + false_pos + "}|{true negative|" + true_neg + "}|{false negative|" + false_neg \

        gstring += "node_" + parent + " -> leaf_" + id + " [label=" + str(int(left)) + "];\n"
    return gstring


def export_graphviz(clf):
    if clf.is_fitted_ is False:  # fit method has not been called
        raise NotFittedError("Call fit method first" % {'name': type(clf).__name__})

    if clf.tree_ is None:
        raise TreeNotFoundError("predict(): ", "Tree not found during training by DL8.5 - "
                                               "Check fitting message for more info.")

    if hasattr(clf, 'tree_') is False:  # normally this case is not possible.
        raise SearchFailedError("PredictionError: ", "DL8.5 training has failed. Please contact the developers "
                                                     "if the problem is in the scope supported by the tool.")

    # initialize the header
    graph_string = "digraph Tree { \n" \
                   "graph [ranksep=0]; \n" \
                   "node [shape=record]; \n"

    # build the body
    graph_string += get_dot_body(clf.tree_)

    # end by the footer
    graph_string += "}"

    return graph_string

# %%
