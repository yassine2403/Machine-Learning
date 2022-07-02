import numpy as np
def covariance(m1,m2):
    return np.cov(m1,m2)
arr=np.array([[1,2,3],[6,7,9]])
arr1=np.array([[1,8,3],[6,8,9]])
print(covariance(arr,arr1))
