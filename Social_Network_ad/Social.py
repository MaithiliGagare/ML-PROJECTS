import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix,accuracy_score,precision_score,recall_score
from sklearn.linear_model import LogisticRegression

df=pd.read_csv(r"C:\Users\maith\OneDrive\Documents\Social_Network_Ads.csv")

print(df['Gender'].replace({'Male':0,'Female':1},inplace=True))
print(df)

print(df.columns)

x=df[['User ID', 'Gender', 'Age', 'EstimatedSalary']]
print(x)

y=df['Purchased']
print(y)

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25,random_state=29)

model=LogisticRegression()
model.fit(x_train,y_train)

y_predi=model.predict(x_test)
print(y_predi)

model.score(x_train,y_train)
model.score(x_test,y_test)

cm=confusion_matrix(y_test,y_predi)
print(cm)

acc=accuracy_score(y_test,y_predi)
print(acc)

tp,fp,tn,fn=confusion_matrix(y_test,y_predi).ravel()
print(tp,fp,tn,fn)

e=1-acc
print(e)
