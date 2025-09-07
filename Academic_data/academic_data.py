import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from sklearn import preprocessing

df=pd.read_csv(r"C:\Users\maith\OneDrive\Documents\acdemic_data.csv")

               
df["Gender"].replace({'M':0,'F':1},inplace=True)

# mean

print(df['DSBDA'].mean())
print(df.mean(axis=1,numeric_only=True)[0:4])


# median

print(df['WT'].median())
print(df.median(axis=1,numeric_only=True)[0:4])

#mode

print(df["DSBDA"].mode())
print(df.mode(axis=1,numeric_only=True)[0:5])

print(df.isnull().sum())

print(df.dropna())

# minimum

print(df["WT"].min())
print(df.min(numeric_only=True))

# Maximum
print(df["DSBDA"].max())
print(df.max(numeric_only=True))

# standard deviation

print(df["DSBDA"].std())
print(df.std(numeric_only=True).std())

#group by

group= print(df.groupby(['DSBDA'])['WT'].mean())
print(group)

print(df.describe())

print(df["DSBDA"].describe().sum())

print(df.groupby('Gender').describe().sum())