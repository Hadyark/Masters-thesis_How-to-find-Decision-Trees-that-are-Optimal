

1-  This directory contains three  datasets, i.e., Census Income dataset, Communities and Crimes dataset
and  German credit dataset available in "A. Asuncion and D. Newman. UCI machine learning repository.
2007  http://archive.ics.uci.edu/ml/datasets.html ". These datasets are provided in ARFF format which can
be used directly in Weka.

2- For these datasets, dependency parameters for the datasets in this directory are kept as follows:

<Sensitve Attribute>
         Sensitive Attribute is an attribute on which class labels are dependent, e.g., gender, race
	   is positioned as a first attribute in these datasets (default position of SA).

 <Desired class index>
         Desired class is class from which the deprived community is denied like Job=yes. Second value
	   of class attribute is selected as desired class.

<Deprived community>
         Value of deprived community of sensitive attribute (SA), e.g., female, black
          First value of SA, i.e., at index 0 of SA is used as Derived Community.
<Favored community>
         Value of favored community of sensitive attribute (SA), e.g., male, white
         Second value of SA, i.e., at index 1 of SA is used as Derived Community.


NOTE: If you want use any other dataset for Weka_with_ICs GUI, please apply the above mentioned default settings on the
used dataset , e.g., position the SA on the first attribute position of the dataset. Otherwise you can specify these
options with CLI commands.
like

java -cp ..\..\weka_with_ICs.jar  weka.indConstraints.ClassiferWithICs -t  ..\..\datasets\census_income.arff  -v
				-F weka.indConstraints.NoFilter -W weka.classifiers.trees.J48 -Z true -C 1 -S 0 -D Female -G Male

3: Explanation of options to specify the dependency parameters is as follows:

-F <filter specification>
        Full class name of preprocessing technique (filter to use) , followed
        by filter options.
        eg: "weka.filters.AttributeFilter -V -R 1,2"
-C <Desired class index>
         Desired class index
        (default: second value of class attribute, i.e., at index 1 of class).).


-S <Sensitve Attribute index>
         Sensitive attribute (SA) index (default: first attribute)

-D <Deprived community>
         Value of deprived community of sensitive attribute (SA)
        (default: first value of SA, i.e., at index 0 of SA).

-G <Favored community>
         Value of favored community of sensitive attribute (SA)
        (default: second value of SA, i.e., at index 1 of SA).

-Z <SAabsent>
        If flag is true, the SA is only used
         for dependency calculation, otherwise SA is a part of
        whole learning and testing process (default: false).

-R <Ranker>
         Complete name of the ranker used with Massaging filter
        (default: weka.classifiers.bayes.NaiveBayesSimple).
-cp <class path>
