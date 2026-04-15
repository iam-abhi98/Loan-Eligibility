import pandas as pd 
import numpy as np
pd.set_option('display.max_columns',None)
df=pd.read_csv('loan_elegibility.csv')
print(df.head())
print(df.info())
print("\nColumns")
print(df.columns)
print(df.isnull().sum())

#Fill categorical columns with mode
df['Gender']=df['Gender'].fillna(df['Gender'].mode()[0])
df['Married']=df['Married'].fillna(df['Married'].mode()[0])
df['Self_Employed']=df['Self_Employed'].fillna(df['Self_Employed'].mode()[0])
#df[''].fillna(df[''].mode()[0],inplace=True)
print(df.isnull().sum())


df['Dependents']=df['Dependents'].replace('3+',3)
df['Dependents']=pd.to_numeric(df['Dependents'])
df['Dependents']=df['Dependents'].fillna(df['Dependents'].median())

df['LoanAmount']=pd.to_numeric(df['LoanAmount'])
df['LoanAmount']=df['LoanAmount'].fillna(df['LoanAmount'].median())

df['Loan_Amount_Term']=pd.to_numeric(df['Loan_Amount_Term'])
df['Loan_Amount_Term']=df['Loan_Amount_Term'].fillna(df['Loan_Amount_Term'].median())

df['Credit_History']=pd.to_numeric(df['Credit_History'])
df['Credit_History']=df['Credit_History'].fillna(df['Credit_History'].median())
print(df.isnull().sum())

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
cols=['Gender','Married','Education','Self_Employed','Property_Area','Loan_Status']
for col in cols:
    df[col]=le.fit_transform(df[col])

print(df.head())

import matplotlib.pyplot as plt
import seaborn as sns 

sns.countplot(x='Loan_Status',data=df)
plt.title('Loan Status Count')
plt.show()

sns.countplot(x='Gender', hue='Loan_Status', data=df)
plt.show()

df['ApplicantIncome'].hist(bins=20)
plt.show()

df['ApplicantIncome']=np.log1p(df['ApplicantIncome'])
df['ApplicantIncome'].hist(bins=20)
plt.show()

sns.boxplot(x=df['LoanAmount'])
plt.show()

df['LoanAmount']=np.log1p(df['LoanAmount'])
sns.boxplot(x=df['LoanAmount'])
plt.show()

sns.pairplot(df)
plt.show()