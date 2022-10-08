

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering 
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import scipy.cluster.hierarchy as shc
from sklearn.cluster import KMeans 

dataset=pd.read_csv(r"C:\Users\Fehmi Laourine\Downloads\CC GENERAL.csv")
dataset['MINIMUM_PAYMENTS'].fillna(dataset['MINIMUM_PAYMENTS'].mean(),inplace=True)
dataset['CREDIT_LIMIT'].fillna(dataset['CREDIT_LIMIT'].mean(),inplace=True)
dataset.drop('CUST_ID',axis=1,inplace=True)



# Standardize data
scaler = StandardScaler() 
scaled_df = scaler.fit_transform(dataset) 
  
# Normalizing the Data 
normalized_df = normalize(scaled_df) 
  
# Converting the numpy array into a pandas DataFrame 
normalized_df = pd.DataFrame(normalized_df) 
  
# Reducing the dimensions of the data 
pca = PCA(n_components = 2) 
X_principal = pca.fit_transform(normalized_df) 
X_principal = pd.DataFrame(X_principal) 
X_principal.columns = ['P1', 'P2'] 


plt.scatter(X_principal['P1'], X_principal['P2'],  
           c = AgglomerativeClustering(n_clusters = 3).fit_predict(X_principal), cmap =plt.cm.winter) 
plt.show() 


plt.figure(figsize =(6, 6)) 
plt.title('Visualising the data') 
Dendrogram = shc.dendrogram((shc.linkage(X_principal, method ='ward'))) 

 
kmeans=KMeans(n_clusters=5, random_state=0)  
kmeans.fit(dataset)


SUMsqrtdists=[]
K=range(1,15)
for k in K:
    km=KMeans(n_clusters=k)
    km=km.fit(dataset)
    SUMsqrtdists.append(km.inertia_)
fig1, ax1 = plt.subplots()
ax1.plot(K,SUMsqrtdists,'bx-')


#best k value is 3
km=KMeans(n_clusters=3)
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=300, c='red', label = 'Centroids')
