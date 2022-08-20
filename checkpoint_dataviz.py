import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.preprocessing import StandardScaler

df=pd.read_csv('C:\\Users\\Fehmi Laourine\\Downloads\\aa.csv',encoding='ISO-8859-1')
df['age'].fillna(df['age'].mean(),inplace=True)
df['cabin'].fillna('G6',inplace=True)
df['home.dest'].fillna(method='bfill',inplace=True)
df = df[df['home.dest'].notna()]
df['embarked'].fillna(method='bfill',inplace=True)
df['fare'].fillna(df['fare'].mode(),inplace=True)
df = df[df['fare'].notna()]
df.drop('boat',axis=1,inplace=True)
df.drop('body',axis=1,inplace=True)


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
    #we create a heatmap with the correlation matrix 
plot_correlation_map(df)
fg=sns.FacetGrid(df,col='survived')
fg.map(plt.hist,'sex',bins=20)
#females appear to be more likely to survive
sns.displot(df['age'], bins=10,kde=True,color='blue')
sns.displot(df['pclass'], bins=10,kde=True,color='red')


fg=sns.FacetGrid(df,col='survived')
fg.map(plt.hist,'age',bins=20)
#age is not correlated with survival
fg=sns.FacetGrid(df,col='survived')
fg.map(plt.hist,'pclass',bins=20)

fg=sns.FacetGrid(df,col='survived')
fg.map(plt.hist,'fare',bins=20)
