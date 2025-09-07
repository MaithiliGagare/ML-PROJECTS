import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

df=pd.read_csv(r"C:\Users\maith\Downloads\boston_housing.csv")
print(df)

print(df.columns)

x=df[['crim', 'zn', 'indus', 'chas', 'nox', 'rm', 'age', 'dis', 'rad', 'tax',
       'ptratio', 'black', 'lstat']]
print(x)

y=df['price']
print(y)

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25,random_state=42)

model=LinearRegression()
model.fit(x_train,y_train)

y_predi=model.predict(x_test)
print(y_predi)

model.score(x_train,y_train)
model.score(x_test,y_test)

np.sqrt(mean_squared_error(y_test,y_predi))

plt.scatter(y_test,y_predi)
plt.plot(y_test,y_predi)
plt.show()
