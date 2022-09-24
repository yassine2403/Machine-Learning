import matplotlib.pyplot as plt
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures 
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import mean_squared_error
from sklearn import metrics

from sklearn.metrics import mean_squared_error
from sklearn import metrics

df=pd.read_csv(r'C:\Users\Fehmi Laourine\Desktop\gomycode\train.csv',encoding='ISO-8859-1')
print(df.isnull().sum())
df['Age'].fillna(df['Age'].mean(),inplace=True)
print(df.columns)
x = df[['Pclass','Age','Parch','Fare']]
y = df['Survived']
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.25,random_state=0)


logreg=LogisticRegression()
logreg.fit(x_train,y_train)
y_pred=logreg.predict(x_test)

print(f'accuracy={logreg.score(x_test, y_test)}')



sns.regplot(x='Age',y='Survived',data=df)
confusion_matrix = pd.crosstab(y_test, y_pred, rownames=['Actual'], colnames=['Predicted'])
sns.heatmap(confusion_matrix, annot=True)
