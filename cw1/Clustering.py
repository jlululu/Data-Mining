# -*- coding: utf-8 -*-
import pandas as pd
import matplotlib.pyplot as plt
import sklearn.cluster as cluster

# 2. Clustering
# 2.1
df = pd.read_csv('wholesale_customers.csv')
df = df.drop(['Channel','Region'],axis=1)
for i in range(6):
    print('the mean of {} = {}'.format(df.columns[i],df.iloc[:,i].mean())) #mean
    print('the range of {} = ({} , {})'.format(df.columns[i],df.iloc[:,i].min(),df.iloc[:,i].max())) #range
    
# 2.2
k = 3
km = cluster.KMeans(n_clusters=k)
km.fit(df.values)
for i in range(len(df.columns)):
    for j in range(i + 1,len(df.columns)):
        plt.scatter(df.iloc[:,i],df.iloc[:,j],c=km.labels_)
        plt.xlabel(df.columns[i])
        plt.ylabel(df.columns[j])
        plt.title('{} vs {}'.format(df.columns[i],df.columns[j]))
        plt.show()

# 2.3
k_list = [3,5,10]
for k in k_list:
    km = cluster.KMeans(n_clusters=k)
    km.fit(df.values)
    WC = km.inertia_
    BC = 0
    cluster_centers = km.cluster_centers_
    for i in range(len(cluster_centers)):
        for j in range(i + 1,len(cluster_centers)):
            BC += sum((cluster_centers[i]-cluster_centers[j])**2)
    print('For K = {}: BC = {}, WC = {}, BC/WC = {}'.format(k,BC,WC,BC/WC))