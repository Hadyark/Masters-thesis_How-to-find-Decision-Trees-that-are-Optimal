# Masters-thesis_How-to-find-Decision-Trees-that-are-Optimal

Decision trees are one of the most popular types of predictive models in machine learning: they are not only sufficiently accurate in many applications, but also provide models that are interpretable. 


However, the most common algorithms for finding decision trees are heuristic in nature: they use information gain as a heuristic for constructing the tree top-down. As a result, the trees that are found using such algorithms may not always be the most accurate on training data and may not always be the smallest.


An alternative approach is to use exhaustive search to find trees that are optimal under well-defined constraints and optimization criteria. A large number of such algorithms has recently been proposed by researchers both in academia and in industry. 


However, we have recently demonstrated that an algorithm developed in our lab, called DL8.5, obtains much better performance than other methods. We have made this algorithm freely available at https://github.com/aglingael/dl8.5/. 


We are interested in extending this algorithm further. 


Possible extensions are:

    We would like to integrate DL8.5 in toolkits such as R;
    We would like to study its performance on multi-label prediction problems, where we wish to predict multiple labels at the same time;
    We would like to improve how the algorithms deals with numerical data;
    We would like to make the algorithm even more efficient;
    We would like to extend the type of constraints supported in our tool, for instance, taking into account fairness.


Within this project you will work on one such extension of DL8.5.


The source code of our system is in a combination of Python and C++, so experience with both these languages is beneficial.


Siegfried Nijssen, Élisa Fromont: Mining optimal decision trees from itemset lattices. KDD 2007: 530-539


Frederic Don-de-dieu Aglin, Siegfried Nijssen, and Pierre Schaus. Learning optimal decision trees using caching branch-and-bound search. In AAAI. 2020.


https://dl85.readthedocs.io/en/latest/
