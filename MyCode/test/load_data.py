import pandas as pd
import numpy as np
from scipy.io import arff


def lawsuit():

    df = pd.read_csv("/home/hadyak/MEGAsync/Master's thesis/Masters-thesis_How-to-find-Decision-Trees-that-are-Optimal/MyCode/dataset_perso/Lawsuit.csv")
    del df['ID']

    salary_mean = np.mean(np.array(df["Sal94"].tolist(), float))
    for i in range(0,len(df)):
        if df.at[i, "Sal94"] >= salary_mean:
            df.at[i, "Salary_mean"] = 1
        else:
            df.at[i, "Salary_mean"] = 0
    del df["Sal94"]
    del df["Sal95"]

    col = ['Rank', 'Dept']
    df = df.drop(['Prate', 'Exper'], axis=1)
    df = pd.get_dummies(df, columns=col)

    for i in range(0,len(df)):
        if df.at[i, "Gender"] == 1:
            df.at[i, "Gender"] = 0
        else:
            df.at[i, "Gender"] = 1

    X = df.loc[:, ~df.columns.isin(['Gender', 'Salary_mean'])]
    y = df['Salary_mean']
    sensitive = df['Gender']
    
    return X, y, sensitive

def census():

    data = arff.loadarff('../datasets2/census_income.arff')
    df = pd.DataFrame(data[0])
    str_df = df.select_dtypes([np.object])
    str_df = str_df.stack().str.decode('utf-8').unstack()
    for col in str_df:
        df[col] = str_df[col]
    for i in range(0,len(df)):
        if str(df.at[i, "sex"]) == "Male":
            df.at[i, "sex"] = 0
        elif str(df.at[i, "sex"]) == "Female":
            df.at[i, "sex"] = 1

        if str(df.at[i, "income_class"]) == "<=50K":
            df.at[i, "income_class"] = 0
        elif str(df.at[i, "income_class"]) == ">50K":
            df.at[i, "income_class"] = 1

    for col in df.columns:
        df = df[ df[col] != "?" ]

    col = ['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race', 'native-country']
    df = df.drop(['education-num', 'age', 'capital-gain', 'capital-loss', 'hours-per-week'], axis=1)
    #df = df.drop(['native-country'], axis=1)
    df = pd.get_dummies(df, columns=col, drop_first=True)
    df.reset_index(inplace = True)
    del df['index']
    
    X = df.loc[:, ~df.columns.isin(['income_class', 'sex'])]
    y = pd.to_numeric(df['income_class'])
    sensitive = df['sex']
    
    return X, y, sensitive