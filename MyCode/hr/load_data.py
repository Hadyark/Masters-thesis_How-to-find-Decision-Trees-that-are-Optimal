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

def hr():
    df = pd.read_csv("../dataset_perso/HRDataset_v14.csv")
    del df['Employee_Name']
    del df['EmpID']
    salary_mean = np.mean(np.array(df["Salary"].tolist(), float))

    for i in range(0,len(df)):
        if df.at[i, "Salary"] >= salary_mean:
            df.at[i, "Salary_mean"] = 1
        else:
            df.at[i, "Salary_mean"] = 0
    del df["Salary"]

    for i in range(0,len(df)):
        df.at[i, "Absences"] = df.at[i, "Absences"] / 5
    for i in range(0,len(df)):
        df.at[i, "EngagementSurvey"] = int(df.at[i, "EngagementSurvey"])
    for i in range(0,len(df)):
        if df.at[i, "HispanicLatino"] == 'Yes' or df.at[i, "HispanicLatino"] == 'yes' :
            df.at[i, "HispanicLatino"] = 1
        elif df.at[i, "HispanicLatino"] == 'No' or df.at[i, "HispanicLatino"] == 'no' :
            df.at[i, "HispanicLatino"] = 0
    col = ['EmpStatusID', 'PerfScoreID', 'Position', 'MaritalDesc', 'CitizenDesc', 'RaceDesc', 'Department', 'PerformanceScore', 'EmpSatisfaction', 'Absences']
    df = df.drop(['MarriedID', 'MaritalStatusID', 'Zip', 'DOB', 'Sex', 'DateofHire','DateofTermination', 'TermReason', 'EmploymentStatus', 'ManagerName', 'ManagerID', 'EngagementSurvey', 'LastPerformanceReview_Date', 'DaysLateLast30', 'RecruitmentSource', 'State', 'DeptID', 'PositionID', 'SpecialProjectsCount'], axis=1)
    df = pd.get_dummies(df, columns=col)
    for col in df:
        if len(df[col].unique()) > 2:
            print(f'{col}: {df[col].unique()}')
    for i in range(0,len(df)):
        if df.at[i, "GenderID"] == 1:
            df.at[i, "GenderID"] = 0
        else:
            df.at[i, "GenderID"] = 1
    X = df.loc[:, ~df.columns.isin(['Gender', 'Salary_mean'])]
    y = df['Salary_mean']
    sensitive = df['GenderID']
    return X, y, sensitive
