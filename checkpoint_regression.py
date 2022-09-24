import matplotlib.pyplot as plt
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures 

from sklearn.metrics import mean_squared_error
from sklearn import metrics

from sklearn.metrics import mean_squared_error
from sklearn import metrics

df=pd.read_csv(r'C:\Users\Fehmi Laourine\Desktop\gomycode\kc_house_data.csv',encoding='ISO-8859-1')
print(df.isnull().sum())

def plot_correlation_map( df ):

    corr = df.corr()
    #we create a matrix of the correlations between the coloumns
    s , ax = plt.subplots( figsize =( 12 , 10 ) )
    #we create a plot s and an axis ax
    cmap = sns.diverging_palette( 200 ,12  , as_cmap = True )
    # we create a colormap ranging from color 200 to 12
    s = sns.heatmap(
        corr, 
        cmap = cmap,
        square=True, 
        cbar_kws={ 'shrink' : .9 }, 
        ax=ax, 
        annot = True, 
        annot_kws = { 'fontsize' : 12 }
        )
plot_correlation_map(df)
#we will take all the features that have a correlation of more than 0.3 with the price
x=df[["sqft_living","grade","sqft_above","bedrooms",'sqft_basement','sqft_living15','lat','view','floors','waterfront']]
y=df["price"]

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25,random_state=40) 

model=LinearRegression()   
model.fit(x_train,y_train)  
predicted=model.predict(x_test)

#we will only plot the sqft_living feature
plt.scatter(x['sqft_living'],y,color="r")
plt.title("Linear Regression")
plt.ylabel("price")
plt.xlabel("sqft_living")
plt.plot(x,model.predict(x),color="b")





lg=LinearRegression()
poly=PolynomialFeatures(degree=2)

x_train_fit = poly.fit_transform(x_train) #
lg.fit(x_train_fit, y_train)
x_test_ = poly.fit_transform(x_test)
predicted = lg.predict(x_test_)
print("MSE", mean_squared_error(y_test,predicted))
print("R squared", metrics.r2_score(y_test,predicted))

#this is the most accurate model
plt.show()
