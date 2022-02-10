import uuid

from dl85.errors.errors import TreeNotFoundError, SearchFailedError
from sklearn.exceptions import NotFittedError


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
        misclassified = str(int(treedict["misclassified"])) if treedict["misclassified"] - int(treedict["misclassified"]) == 0 else str(
            round(treedict["misclassified"], 3))
        discr = str(int(treedict["discrimination_additive"])) if treedict["discrimination_additive"] - int(treedict["discrimination_additive"]) == 0 else str(
            round(treedict["discrimination_additive"], 3))
        # maxi = max(len(val), len(err))
        # val = val if len(val) == maxi else val + (" " * (maxi - len(val)))
        # err = err if len(err) == maxi else err + (" " * (maxi - len(err)))
        gstring += "leaf_" + id + " [label=\"{{class|" + val + "}|{error|" + err + "}|{misclassified|" + misclassified + "}|{discrimination|" + discr + "}}\"];\n"
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
