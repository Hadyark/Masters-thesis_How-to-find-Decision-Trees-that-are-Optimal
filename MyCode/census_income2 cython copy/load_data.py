import pandas as pd
import numpy as np
from scipy.io import arff


def lawsuit():

    df = pd.read_csv("../dataset_perso/Lawsuit.csv")
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


    print(df.min())
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
            
        if df.at[i, 'hours-per-week'] < 10:
            df.at[i, 'hours-per-week'] = 10
        elif df.at[i, 'hours-per-week'] <20:
            df.at[i, 'hours-per-week'] = 20
        elif df.at[i, 'hours-per-week'] <30:
            df.at[i, 'hours-per-week'] = 30
        elif df.at[i, 'hours-per-week'] <40:
            df.at[i, 'hours-per-week'] = 40
        elif df.at[i, 'hours-per-week'] <50:
            df.at[i, 'hours-per-week'] = 50
        elif df.at[i, 'hours-per-week'] <60:
            df.at[i, 'hours-per-week'] = 60
        elif df.at[i, 'hours-per-week'] <70:
            df.at[i, 'hours-per-week'] = 70
        elif df.at[i, 'hours-per-week'] <80:
            df.at[i, 'hours-per-week'] = 80
        elif df.at[i, 'hours-per-week'] <90:
            df.at[i, 'hours-per-week'] = 90
        elif df.at[i, 'hours-per-week'] < 100:
            df.at[i, 'hours-per-week'] = 100
            
        if df.at[i, 'capital-gain'] < 1000:
            df.at[i, 'capital-gain'] = 1000
        elif df.at[i, 'capital-gain'] <2000:
            df.at[i, 'capital-gain'] = 2000
        elif df.at[i, 'capital-gain'] <3000:
            df.at[i, 'capital-gain'] = 3000
        elif df.at[i, 'capital-gain'] <4000:
            df.at[i, 'capital-gain'] = 4000
        elif df.at[i, 'capital-gain'] <5000:
            df.at[i, 'capital-gain'] = 5000
        elif df.at[i, 'capital-gain'] <6000:
            df.at[i, 'capital-gain'] = 6000
        elif df.at[i, 'capital-gain'] <7000:
            df.at[i, 'capital-gain'] = 7000
        elif df.at[i, 'capital-gain'] <8000:
            df.at[i, 'capital-gain'] = 8000
        elif df.at[i, 'capital-gain'] <9000:
            df.at[i, 'capital-gain'] = 9000
        elif df.at[i, 'capital-gain'] <10000:
            df.at[i, 'capital-gain'] = 10000

        if df.at[i, 'capital-loss'] < 1000:
            df.at[i, 'capital-loss'] = 1000
        elif df.at[i, 'capital-loss'] <2000:
            df.at[i, 'capital-loss'] = 2000
        elif df.at[i, 'capital-loss'] <3000:
            df.at[i, 'capital-loss'] = 3000
        elif df.at[i, 'capital-loss'] <4000:
            df.at[i, 'capital-loss'] = 4000
            
    for col in df.columns:
        df = df[ df[col] != "?" ]

    col = ['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race', 'native-country', 'hours-per-week' , 'capital-gain', 'capital-loss']
    df = df.drop(['education-num', 'age'], axis=1)
    #df = df.drop(['native-country'], axis=1)
    df = pd.get_dummies(df, columns=col, drop_first=True)
    df.reset_index(inplace = True)
    del df['index']
    
    X = df.loc[:, ~df.columns.isin(['income_class', 'sex'])]
    y = pd.to_numeric(df['income_class'])
    sensitive = df['sex']
    
    return X, y, sensitive